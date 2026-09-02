"""Independent self-consistency check for the uni_v1 Macenko corpus.

Agreement with the wsi_preprocess side (which runs its own
`run_check_consistency.sh` as the gate before FAISS build): patch-vector-search
independently recomputes the same quantity and both sides must see cos_self ~1.0
before experiments/0012 (build_faiss_index) is run.

What "self-consistency" means here: take a patch that is already in the delivered
Macenko corpus (its feature vector is stored in
data/trident_processed_uni_v1_macenko/.../features_uni_v1_macenko/<slide>.h5),
re-crop it fresh from the raw WSI at its stored level-0 coordinates, run the
*same* pipeline the corpus was built with (torchstain NumpyMacenkoNormalizer
toward data/baseline/63958_x38976_y7616.png, then uni_v1 with TRIDENT's
eval_transforms), and compare the resulting embedding to the stored one by
cosine. If the pipeline is faithfully reproduced this is ~1.0 (it IS ~1.0 for
the plain uni_v1 corpus via the same test; lib.stain_normalize's from-scratch
Macenko only reached 0.5-0.84, which is why lib.torchstain_normalize exists —
see its docstring).

cos_vs_old is a secondary column: cosine between the new (Macenko) stored
feature and the OLD non-normalized corpus feature
(data/trident_processed/.../features_uni_v1/<slide>.h5) at the same location.
It is NOT expected to be ~1.0 — it measures how far Macenko moved the
embedding. The Macenko corpus was re-segmented, so patch sets do not line up
1:1 with the old corpus; when there is no exact coordinate match the nearest
old-corpus patch is used and its coord distance is recorded (old_match_dist).

Resampling-path caveat for patch_size_level0 != 224 slides (40x-native): TRIDENT
extracts by reading a chosen pyramid LEVEL (get_best_level_and_custom_downsample,
not always level 0) at patch_size_level and then PIL-resizing to 224 with the
default BICUBIC filter, before Macenko. This script instead reads level 0 at
patch_size_level0 and resizes with LANCZOS (via lib.raw_patch.crop_patch, matching
the wsi_preprocess audit scripts so the two independent checks agree). For
ps0 == 224 (20x-native, the bulk of the TG-GATE rat-liver corpus) no resize
happens and the paths are identical — that subset is the real gate. For
ps0 != 224 a few thousandths of cos_self can be lost to the resampling-path
difference alone (nothing to do with Macenko); that subset is reported
separately and only advised on, not gated.

Usage (from PROJECT_ROOT, needs the uni_v1 encoder weights; GPU optional):
    .venv/bin/python3 scripts/stain_norm_self_consistency_diagnostic.py
    .venv/bin/python3 scripts/stain_norm_self_consistency_diagnostic.py --n-slides 1  # smoke test

Degenerate-normalization note (2026-09-01, from a wsi_preprocess log audit):
torchstain's NumpyMacenkoNormalizer does NOT raise on a patch it cannot
normalize (near-background / faded patches where the OD covariance degenerates).
It runs to completion with NaN/inf intermediates and returns an essentially
all-white (overflow -> 255 clip) or all-black (NaN -> 0 cast) patch. So the
corpus's own "exception -> passthrough raw" path never fires.

Failure detection is split to match the wsi_preprocess audit
(audit_stain_norm_failures.py):
  - PRIMARY (`stain_norm_failed`, the count that feeds fail_rate / manifest
    exclusion): normalization run with RuntimeWarning promoted to an error and
    NumPy divide/over/invalid set to raise (under left ignored — np.exp
    underflow fires on healthy 40x / densely-stained patches too) tripped, or an
    exception was raised. A rank-deficient OD covariance ("Degrees of freedom
    <= 0"), "overflow in exp", "invalid value in cast" are unambiguous signals
    that the stain-vector estimation broke down; healthy tissue patches never
    trigger them. The returned patch is always finite (torchstain clips + casts
    before returning), so the output alone can't be checked — the warning is the
    signal. This is exactly wsi_preprocess's try_normalize(strict=True); the
    corpus's Step3 still uses the non-strict path, so the degenerate white/black
    patch is what actually got embedded (and what this script re-embeds).
  - ADVISORY (`collapsed_output_advisory`, NOT part of fail_rate): the raw crop
    had real texture but the normalized output collapsed to a near-constant
    frame with no warning — the borderline case (a handful of tissue pixels
    keeping the covariance barely non-degenerate). Surfaced separately so both
    sides can see whether this population is large enough to promote.

Patches flagged PRIMARY embed to near-identical white/black vectors on both the
corpus and the re-crop, so cos_self stays ~1.0 for them — the flag, not
cos_self, is what surfaces them. They should also be in stain_norm_failures.json
and thus excluded from the manifest; pass --failures-json to skip them when
sampling.

Output: outputs/stain_norm_self_consistency.csv
    columns: slide_id,row_index,coord_x,coord_y,patch_size_level0,
             stain_norm_failed,stain_norm_fail_reason,collapsed_output_advisory,
             cos_self,cos_vs_old,old_match_dist
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd

# Defaults are PROJECT_ROOT-relative; the script must be run from PROJECT_ROOT
# (same convention as the other scripts/*_diagnostic.py).
MACENKO_FEATURES_DIR = "data/trident_processed_uni_v1_macenko/20x_224px_0px_overlap/features_uni_v1_macenko"
# The wsi_preprocess finalize step renames features_uni_v1 -> features_uni_v1_macenko.
# Before that rename the same files live here; used as a fallback so the smoke
# test works against partial output.
MACENKO_FEATURES_DIR_PRERENAME = "data/trident_processed_uni_v1_macenko/20x_224px_0px_overlap/features_uni_v1"
OLD_FEATURES_DIR = "data/trident_processed/20x_224px_0px_overlap/features_uni_v1"
RAW_WSI_DIR = "data/moo_collected_tggate_wsi/raw_wsi"
STAIN_REFERENCE = "data/baseline/63958_x38976_y7616.png"
OUTPUT_CSV = "outputs/stain_norm_self_consistency.csv"

N_SLIDES = 20
N_PATCHES_PER_SLIDE = 5
SEED = 42
# cos_self below this for any sampled patch fails the gate. 0.98 mirrors the
# margin the plain-corpus / torchstain checks clear comfortably (0.96-0.996);
# anything materially below that means the reproduced pipeline diverges.
COS_SELF_GATE = 0.98

# ADVISORY (not primary) detection: raw crop had texture, normalized output
# collapsed to a near-constant frame with no warning. Matches
# wsi_preprocess's audit_stain_norm_failures.py advisory thresholds.
DEGENERATE_STD = 1.0    # normalized output std below this = collapsed
RAW_TISSUE_STD = 5.0    # raw crop std above this = there was real content to lose


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _strict_stain_norm_failed(normalize_fn, crop, reference) -> str:
    """PRIMARY failure check, mirroring wsi_preprocess's
    MacenkoStainNormalizer.try_normalize(strict=True): run normalization with
    RuntimeWarning promoted to an error and NumPy divide/over/invalid set to
    raise (under stays ignored — np.exp underflow happens on healthy patches
    too, esp. 40x / densely stained, and would false-positive). Returns a short
    reason string if it trips, "" otherwise. Does NOT mutate what gets embedded:
    the corpus's Step3 uses the non-strict path and embeds the degenerate
    white/black frame, so the caller re-runs normalization without this guard
    for the actual embedding.
    """
    try:
        with warnings.catch_warnings(), np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
            warnings.simplefilter("error", RuntimeWarning)
            normalize_fn(crop, reference)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:120]
    return ""


def _collapsed_without_warning(raw_crop, normed) -> bool:
    """ADVISORY: the raw crop had real texture but the normalized output
    collapsed to a near-constant frame — the borderline case that does not
    raise a warning (see module docstring). Not part of the primary failure
    count; surfaced separately.
    """
    out_std = float(np.asarray(normed, dtype=np.float32).std())
    raw_std = float(np.asarray(raw_crop, dtype=np.float32).std())
    return out_std < DEGENERATE_STD and raw_std > RAW_TISSUE_STD


def _resolve_features_dir(project_root: Path, cli_value: str | None) -> Path:
    if cli_value:
        d = project_root / cli_value
        if not d.is_dir():
            raise FileNotFoundError(f"--features-dir not found: {d}")
        return d
    d = project_root / MACENKO_FEATURES_DIR
    if d.is_dir() and any(d.glob("*.h5")):
        return d
    fallback = project_root / MACENKO_FEATURES_DIR_PRERENAME
    if fallback.is_dir() and any(fallback.glob("*.h5")):
        print(f"NOTE: {MACENKO_FEATURES_DIR} not populated — using pre-rename dir {fallback}")
        return fallback
    raise FileNotFoundError(
        f"no Macenko feature h5 files under {d} or {fallback} — corpus not delivered yet"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", default=None, help="override the Macenko features dir (PROJECT_ROOT-relative)")
    p.add_argument("--n-slides", type=int, default=N_SLIDES)
    p.add_argument("--n-patches", type=int, default=N_PATCHES_PER_SLIDE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: cuda if available)")
    p.add_argument("--out", default=OUTPUT_CSV)
    p.add_argument("--failures-json", default=None,
                   help="stain_norm_failures.json from the audit; its rows are the "
                        "degenerate patches excluded from the manifest — skip them when sampling")
    return p.parse_args()


def _load_failure_rows(path: Path) -> dict[str, set[int]]:
    import json
    payload = json.loads(path.read_text())
    by_slide: dict[str, set[int]] = {}
    for rec in payload.get("failures", []):
        by_slide.setdefault(str(rec["slide_id"]), set()).add(int(rec["row_index"]))
    return by_slide


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    from lib.query_embedding import embed_image
    from lib.raw_patch import crop_patch
    from lib.torchstain_normalize import normalize_to_reference

    features_dir = _resolve_features_dir(project_root, args.features_dir)
    old_features_dir = project_root / OLD_FEATURES_DIR
    raw_wsi_dir = project_root / RAW_WSI_DIR
    stain_reference = project_root / STAIN_REFERENCE
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    failure_rows: dict[str, set[int]] = {}
    if args.failures_json:
        failure_rows = _load_failure_rows(project_root / args.failures_json)
        print(f"loaded {sum(len(v) for v in failure_rows.values())} manifest-excluded "
              f"rows across {len(failure_rows)} slides from {args.failures_json} — skipping them")

    all_h5 = sorted(features_dir.glob("*.h5"))
    # only slides whose raw WSI is present (needed to re-crop)
    candidates = [h for h in all_h5 if (raw_wsi_dir / f"{h.stem}.svs").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"none of the {len(all_h5)} feature h5 files have a matching raw WSI under {raw_wsi_dir}"
        )
    n_slides = min(args.n_slides, len(candidates))
    chosen = [candidates[i] for i in rng.permutation(len(candidates))[:n_slides]]
    print(f"sampling {args.n_patches} patches from each of {n_slides} slides "
          f"({len(candidates)} slides have a raw WSI)")

    rows: list[dict] = []
    for si, h5_path in enumerate(chosen):
        slide_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            feats = f["features"][:].astype(np.float32)
            ps0 = int(round(float(f["coords"].attrs["patch_size_level0"])))
        n_patches = coords.shape[0]

        old_coords = old_feats = None
        old_h5 = old_features_dir / f"{slide_id}.h5"
        if old_h5.exists():
            with h5py.File(old_h5, "r") as f:
                old_coords = f["coords"][:]
                old_feats = f["features"][:].astype(np.float32)

        skip = failure_rows.get(slide_id, set())
        order = [int(r) for r in rng.permutation(n_patches) if int(r) not in skip]
        pick = order[: args.n_patches]
        for row_index in pick:
            row_index = int(row_index)
            cx, cy = int(coords[row_index, 0]), int(coords[row_index, 1])
            stored = _l2(feats[row_index])

            crop = crop_patch(
                slide_id=slide_id,
                coord_x=cx,
                coord_y=cy,
                raw_slide_dir=raw_wsi_dir,
                patch_size_level0=ps0,
                target_size=224,
            )
            # PRIMARY check (strict=True equivalent). Does not affect what we
            # embed — the corpus's Step3 uses the non-strict path.
            fail_reason = _strict_stain_norm_failed(normalize_to_reference, crop, stain_reference)
            stain_norm_failed = bool(fail_reason)

            # Actual embedding: reproduce exactly what the corpus did — non-strict
            # normalize (torchstain runs to completion, returning the white/black
            # frame for a degenerate patch), then uni_v1.
            normed = normalize_to_reference(crop, stain_reference)
            collapsed_advisory = _collapsed_without_warning(crop, normed)
            if stain_norm_failed:
                print(f"  {slide_id} row {row_index}: stain-norm PRIMARY failure ({fail_reason})")
            elif collapsed_advisory:
                print(f"  {slide_id} row {row_index}: collapsed output, no warning (advisory)")

            repro = embed_image(normed, device=args.device, encoder_name="uni_v1", resize_mode="centercrop")
            cos_self = float(np.dot(_l2(repro), stored))

            cos_vs_old = np.nan
            old_match_dist = np.nan
            if old_coords is not None:
                d = np.hypot(old_coords[:, 0] - cx, old_coords[:, 1] - cy)
                j = int(np.argmin(d))
                old_match_dist = float(d[j])
                cos_vs_old = float(np.dot(stored, _l2(old_feats[j])))

            rows.append({
                "slide_id": slide_id,
                "row_index": row_index,
                "coord_x": cx,
                "coord_y": cy,
                "patch_size_level0": ps0,
                "stain_norm_failed": stain_norm_failed,
                "stain_norm_fail_reason": fail_reason,
                "collapsed_output_advisory": collapsed_advisory,
                "cos_self": cos_self,
                "cos_vs_old": cos_vs_old,
                "old_match_dist": old_match_dist,
            })
        if pick:
            batch = [r["cos_self"] for r in rows[-len(pick):]]
            print(f"[{si + 1}/{n_slides}] {slide_id}: cos_self {min(batch):.4f}..{max(batch):.4f}")
            pd.DataFrame(rows).to_csv(out_path, index=False)
        else:
            print(f"[{si + 1}/{n_slides}] {slide_id}: all sampled rows excluded by --failures-json, skipped")

    if not rows:
        print("\nno patches sampled (every candidate row excluded by --failures-json?)")
        sys.exit(1)
    df = pd.DataFrame(rows)

    def _summary(label: str, sub: pd.DataFrame) -> None:
        if not len(sub):
            return
        v = sub["cos_self"].to_numpy()
        print(f"  [{label}] n={len(v)} ({sub['slide_id'].nunique()} slides)  "
              f"min={v.min():.4f}  p05={np.percentile(v, 5):.4f}  median={np.median(v):.4f}  "
              f"mean={v.mean():.4f}  max={v.max():.4f}  below {COS_SELF_GATE}: {int((v < COS_SELF_GATE).sum())}")

    is_20x = df["patch_size_level0"] == 224
    df_20x = df[is_20x]
    df_other = df[~is_20x]
    print("\n=== self-consistency (cos_self) ===")
    # ps0 == 224: identical resampling path, this is the real gate.
    # ps0 != 224: read-level + resize-filter path differs from TRIDENT (see
    # module docstring) — advisory only.
    _summary("ps0==224 / gate", df_20x)
    _summary("ps0!=224 / advisory", df_other)
    n_primary = int(df["stain_norm_failed"].sum())
    n_advisory = int(df["collapsed_output_advisory"].sum())
    if n_primary or n_advisory:
        print(f"\n  stain-norm PRIMARY failures (warning-or-exception): {n_primary}/{len(df)} "
              f"({n_primary / len(df):.2%})")
        print(f"  stain-norm ADVISORY (collapsed output, no warning): {n_advisory}/{len(df)}")
    if df["cos_vs_old"].notna().any():
        vo = df["cos_vs_old"].dropna().to_numpy()
        exact = df.loc[df["cos_vs_old"].notna(), "old_match_dist"].eq(0).sum()
        print("\n=== effect of Macenko vs old non-normalized corpus (cos_vs_old) ===")
        print(f"  n={len(vo)} ({exact} exact-coord matches)  min={vo.min():.4f}  "
              f"median={np.median(vo):.4f}  mean={vo.mean():.4f}  max={vo.max():.4f}")

    print(f"\nwrote {out_path}")

    advisory_fail = df_other[df_other["cos_self"] < COS_SELF_GATE]
    if len(advisory_fail):
        print(f"\nADVISORY — {len(advisory_fail)} ps0!=224 patch(es) below {COS_SELF_GATE} "
              "(expected: resampling-path difference, not Macenko — see module docstring)")

    gate_fail = df_20x[df_20x["cos_self"] < COS_SELF_GATE]
    if len(gate_fail):
        print(f"\nGATE: FAIL — {len(gate_fail)} ps0==224 patch(es) below cos_self {COS_SELF_GATE}:")
        for _, r in gate_fail.iterrows():
            print(f"  {r['slide_id']} row {int(r['row_index'])}: cos_self={r['cos_self']:.4f}"
                  f"{' (stain-norm failed)' if r['stain_norm_failed'] else ''}")
        sys.exit(1)
    if not len(df_20x):
        print("\nGATE: INCONCLUSIVE — no ps0==224 patches in the sample")
        sys.exit(1)
    print(f"\nGATE: PASS — all {len(df_20x)} ps0==224 patches have cos_self >= {COS_SELF_GATE}")


if __name__ == "__main__":
    main()
