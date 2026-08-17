"""Train and populate the cluster-based (OPQ+IVF+PQ) FAISS index over all patch UNI features.

IVF partitions the corpus into Voronoi cells (clusters) so a query only
searches the handful of cells near it instead of all ~19M vectors; PQ
compresses each vector to a few dozen bytes so the whole corpus fits in a
couple of GB instead of the ~78GB raw float32 would need. OPQ learns a
rotation before PQ quantization to reduce the accuracy loss PQ otherwise
causes on high-dimensional (1024-d) vectors like these.
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_faiss_index(
    training_sample_path: str | Path,
    features_dir: str | Path,
    manifest_path: str | Path,
    index_path: str | Path,
    dim: int = 1024,
    nlist: int = 4096,
    pq_m: int = 64,
    pq_nbits: int = 8,
    opq_niter: int = 10,
) -> faiss.Index:
    """Train an OPQ+IVF+PQ index on a sample and add every patch vector to it.

    Vectors are L2-normalized before training/adding so the index's inner
    product metric acts as cosine similarity, matching the convention used
    by lib.query_embedding.embed_image and lib.search.PatchIndex.

    The index is built by hand (OPQMatrix + IndexIVFPQ wrapped in an
    IndexPreTransform) rather than via a bare `faiss.index_factory(...,
    "OPQ64,IVF4096,PQ64x8", ...)` call, specifically so opq_niter is
    reachable: OPQMatrix's default niter=25 was measured (locally, on a
    small sample) to make training time scale roughly with
    n_training_vectors * niter, which is prohibitively slow at millions of
    training vectors. niter=10 trades a bit of OPQ rotation quality for a
    large constant-factor speedup.

    Args:
        training_sample_path: .npy file with a representative sample of raw
            (unnormalized) float32 feature vectors, from build_patch_manifest.
        features_dir: Directory containing {slide_id}.h5 feature files.
        manifest_path: Per-patch manifest (parquet) from build_patch_manifest;
            its global_idx column is used as the FAISS vector ID, so results
            can be joined back to (slide_id, coord_x, coord_y) at query time.
        index_path: Output path for the trained+populated index.
        dim: Embedding dimensionality (1024 for uni_v1).
        nlist: Number of IVF (Voronoi cluster) cells.
        pq_m: Number of PQ subquantizers (must divide dim).
        pq_nbits: Bits per PQ subquantizer code (256 centroids at nbits=8).
        opq_niter: OPQ rotation-refinement iterations (faiss default is 25).

    Returns:
        The trained, populated faiss.Index (also written to index_path).
    """
    if dim % pq_m != 0:
        raise ValueError(f"pq_m ({pq_m}) must divide dim ({dim})")

    faiss.omp_set_num_threads(faiss.omp_get_max_threads())

    training_sample = np.load(training_sample_path).astype(np.float32)
    faiss.normalize_L2(training_sample)

    opq_matrix = faiss.OPQMatrix(dim, pq_m)
    opq_matrix.niter = opq_niter
    quantizer = faiss.IndexFlatIP(dim)
    ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexPreTransform(opq_matrix, ivfpq)

    logger.info(
        "training index on %d vectors (nlist=%d pq_m=%d pq_nbits=%d opq_niter=%d)",
        training_sample.shape[0], nlist, pq_m, pq_nbits, opq_niter,
    )
    index.train(training_sample)
    del training_sample

    manifest = pd.read_parquet(manifest_path, columns=["slide_id", "local_idx", "global_idx"])
    features_dir = Path(features_dir)

    for i, (slide_id, group) in enumerate(manifest.groupby("slide_id", sort=False)):
        group = group.sort_values("local_idx")
        with h5py.File(features_dir / f"{slide_id}.h5", "r") as f:
            vectors = f["features"][:].astype(np.float32)
        if vectors.shape[0] != len(group):
            raise ValueError(
                f"{slide_id}: manifest has {len(group)} rows but h5 has {vectors.shape[0]} features"
            )

        faiss.normalize_L2(vectors)
        ids = group["global_idx"].to_numpy().astype(np.int64)
        index.add_with_ids(vectors, ids)

        if (i + 1) % 100 == 0:
            logger.info("added %d/%d slides (ntotal=%d)", i + 1, manifest["slide_id"].nunique(), index.ntotal)

    faiss.write_index(index, str(index_path))
    logger.info("wrote index to %s (ntotal=%d)", index_path, index.ntotal)
    return index
