"""Query interface over the built FAISS patch index: similar-patch search and WSI reverse lookup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd


@dataclass
class PatchIndex:
    index: faiss.Index
    manifest: pd.DataFrame  # indexed by global_idx
    slide_meta: pd.DataFrame  # indexed by slide_id
    features_dir: Path

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        manifest_path: str | Path,
        slide_meta_path: str | Path,
        features_dir: str | Path,
    ) -> "PatchIndex":
        index = faiss.read_index(str(index_path))
        manifest = pd.read_parquet(manifest_path).set_index("global_idx", drop=False)
        slide_meta = pd.read_parquet(slide_meta_path).set_index("slide_id")
        return cls(index=index, manifest=manifest, slide_meta=slide_meta, features_dir=Path(features_dir))

    def _set_nprobe(self, nprobe: int) -> None:
        # lib.faiss_index.build_faiss_index wraps the IndexIVFPQ inside an
        # IndexPreTransform (for the OPQ rotation), which has no `nprobe` of
        # its own — extract_index_ivf digs through such wrappers to find the
        # underlying IVF index. Setting self.index.nprobe directly would
        # silently create an unused attribute instead of erroring.
        faiss.extract_index_ivf(self.index).nprobe = nprobe

    def _exact_similarity(self, query_vec: np.ndarray, candidates: pd.DataFrame) -> np.ndarray:
        """Recompute exact cosine similarity for a candidate shortlist.

        FAISS's PQ-compressed distances are approximate; this re-reads raw
        float32 vectors from the source h5 files for just the candidates
        (a small shortlist, so the extra I/O is cheap) and recomputes the
        true inner product against the (already L2-normalized) query_vec.
        `candidates` must have a plain 0..N-1 positional index.
        """
        similarities = np.empty(len(candidates), dtype=np.float32)
        for slide_id, positions in candidates.groupby("slide_id").indices.items():
            local_idx = candidates["local_idx"].to_numpy()[positions]
            # h5py fancy indexing requires strictly increasing indices.
            sort_order = np.argsort(local_idx)
            with h5py.File(self.features_dir / f"{slide_id}.h5", "r") as f:
                vectors = f["features"][local_idx[sort_order]].astype(np.float32)
            faiss.normalize_L2(vectors)
            sims_sorted = vectors @ query_vec
            sims = np.empty_like(sims_sorted)
            sims[sort_order] = sims_sorted
            similarities[positions] = sims
        return similarities

    def search_similar_patches(
        self,
        query_vec: np.ndarray,
        k: int = 20,
        nprobe: int = 32,
        rerank_pool: int = 200,
    ) -> pd.DataFrame:
        """Find the k patches most similar to query_vec.

        Retrieves an approximate candidate pool from the FAISS IVF+PQ index,
        then exact-reranks it (see _exact_similarity) to counteract PQ's
        quantization error.

        Args:
            query_vec: L2-normalized (1024,) query embedding, e.g. from
                lib.query_embedding.embed_image.
            k: Number of results to return.
            nprobe: Number of IVF clusters to probe (higher = slower, more recall).
            rerank_pool: Size of the approximate candidate pool to exact-rerank
                (must be >= k).

        Returns:
            DataFrame[slide_id, coord_x, coord_y, similarity], sorted by
            similarity descending.
        """
        self._set_nprobe(nprobe)
        query = query_vec.astype(np.float32)
        _, ids = self.index.search(query.reshape(1, -1), rerank_pool)
        ids = ids[0]
        ids = ids[ids >= 0]  # faiss pads with -1 when fewer than rerank_pool are found

        candidates = self.manifest.loc[ids].reset_index(drop=True)
        candidates["similarity"] = self._exact_similarity(query, candidates)
        candidates = candidates.sort_values("similarity", ascending=False).head(k)
        return candidates[["slide_id", "coord_x", "coord_y", "similarity"]].reset_index(drop=True)

    def search_top_slides(
        self,
        query_vec: np.ndarray,
        k_candidates: int = 8000,
        nprobe: int = 64,
        top_n_slides: int = 20,
    ) -> pd.DataFrame:
        """Reverse-lookup: rank WSIs by how many of their patches are similar to query_vec.

        Uses the raw approximate FAISS candidate pool (no exact re-rank)
        since this is an aggregate ranking, not a per-patch precision-critical
        result.

        Args:
            query_vec: L2-normalized (1024,) query embedding.
            k_candidates: Size of the approximate candidate pool to aggregate
                over. Kept large by default so one or two strongly-matching
                slides can't consume the whole pool and starve other
                genuinely relevant slides of any candidates at all.
            nprobe: Number of IVF clusters to probe.
            top_n_slides: Number of top slides to return.

        Returns:
            DataFrame[slide_id, n_hits, mean_similarity, max_similarity,
            total_patches, n_hits_ratio], sorted by n_hits_ratio descending.
            n_hits_ratio (=n_hits/total_patches) is the ranking key rather
            than raw n_hits — measured on a 7-category ground-truth
            comparison (single_finding_liver.csv) to win or tie the best
            rank in 5/7 categories, sometimes by a wide margin (e.g. one
            category's best true-positive rank went from 46 to 20, another's
            181 to 124), at the cost of a small regression in 1/7. Raw
            n_hits is still in the output for reference/tie-breaking, but
            larger slides can accumulate more hits by volume alone, and
            n_hits_ratio corrects for that.
        """
        self._set_nprobe(nprobe)
        query = query_vec.astype(np.float32).reshape(1, -1)
        distances, ids = self.index.search(query, k_candidates)
        distances, ids = distances[0], ids[0]
        valid = ids >= 0
        distances, ids = distances[valid], ids[valid]

        hits = self.manifest.loc[ids, ["slide_id"]].copy()
        hits["similarity"] = distances

        agg = hits.groupby("slide_id")["similarity"].agg(
            n_hits="size", mean_similarity="mean", max_similarity="max"
        )
        agg = agg.join(self.slide_meta[["total_patches"]])
        agg["n_hits_ratio"] = agg["n_hits"] / agg["total_patches"]
        agg = agg.sort_values("n_hits_ratio", ascending=False).head(top_n_slides)
        return agg.reset_index()

    def search_similar_patches_multi(
        self,
        query_vecs: np.ndarray,
        k: int = 20,
        nprobe: int = 32,
        rerank_pool: int = 200,
        max_tiles_reranked: int = 4,
    ) -> pd.DataFrame:
        """search_similar_patches, aggregated over multiple query vectors
        (e.g. the tiles of one large reference image from
        lib.query_embedding.embed_image_tiles / embed_image_tiles_auto_scale).

        Exact-reranking every tile against the full index costs one h5 read
        pass per tile; most tiles of a large reference image don't contain
        the diagnostic feature (see embed_image_tiles's docstring), so this
        first ranks tiles by a cheap raw FAISS score (no h5 reads) and only
        exact-reranks the top `max_tiles_reranked` of them.

        Args:
            query_vecs: (n_tiles, 1024) L2-normalized query embeddings.
            k: Number of results to return.
            nprobe: Number of IVF clusters to probe.
            rerank_pool: Exact-rerank pool size per reranked tile (see
                search_similar_patches).
            max_tiles_reranked: How many top-scoring tiles to exact-rerank.

        Returns:
            DataFrame[slide_id, coord_x, coord_y, similarity], deduplicated
            by (slide_id, coord_x, coord_y) keeping each patch's best
            similarity across tiles, sorted by similarity descending.
        """
        self._set_nprobe(nprobe)
        query_vecs = np.atleast_2d(query_vecs).astype(np.float32)
        approx_scores = self.index.search(query_vecs, 1)[0][:, 0]
        top_tile_idx = np.argsort(approx_scores)[::-1][:max_tiles_reranked]

        frames = [
            self.search_similar_patches(query_vecs[i], k=k, nprobe=nprobe, rerank_pool=rerank_pool)
            for i in top_tile_idx
        ]
        merged = pd.concat(frames, ignore_index=True).sort_values("similarity", ascending=False)
        merged = merged.drop_duplicates(subset=["slide_id", "coord_x", "coord_y"]).head(k)
        return merged.reset_index(drop=True)

    def search_top_slides_multi(
        self,
        query_vecs: np.ndarray,
        k_candidates: int = 8000,
        nprobe: int = 64,
        top_n_slides: int = 20,
    ) -> pd.DataFrame:
        """search_top_slides, aggregated over multiple query vectors (e.g. the
        tiles of one large reference image from lib.query_embedding.embed_image_tiles).

        A slide that matches strongly to ANY one tile should rank highly, not
        just one whose average across all tiles is best — so this searches
        each tile's candidate pool independently and unions them, rather
        than averaging the query vectors into one and doing a single search.
        The same indexed patch can be a near-neighbor of more than one tile;
        it's counted once per slide (keeping its best similarity across
        tiles) so a slide's n_hits reflects breadth of matching content, not
        repeat hits on the same patch.

        Args:
            query_vecs: (n_tiles, 1024) L2-normalized query embeddings.
            k_candidates: Candidate pool size per tile (see search_top_slides).
            nprobe: Number of IVF clusters to probe.
            top_n_slides: Number of top slides to return.

        Returns:
            Same shape as search_top_slides: DataFrame[slide_id, n_hits,
            mean_similarity, max_similarity, total_patches, n_hits_ratio],
            sorted by n_hits_ratio descending (see search_top_slides's
            docstring for why — measured to win/tie 5/7 categories on a
            ground-truth comparison).
        """
        self._set_nprobe(nprobe)
        frames = []
        for query_vec in np.atleast_2d(query_vecs):
            query = query_vec.astype(np.float32).reshape(1, -1)
            distances, ids = self.index.search(query, k_candidates)
            distances, ids = distances[0], ids[0]
            valid = ids >= 0
            frames.append(pd.DataFrame({"global_idx": ids[valid], "similarity": distances[valid]}))

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.groupby("global_idx", as_index=False)["similarity"].max()
        combined["slide_id"] = combined["global_idx"].map(self.manifest["slide_id"])

        agg = combined.groupby("slide_id")["similarity"].agg(
            n_hits="size", mean_similarity="mean", max_similarity="max"
        )
        agg = agg.join(self.slide_meta[["total_patches"]])
        agg["n_hits_ratio"] = agg["n_hits"] / agg["total_patches"]
        agg = agg.sort_values("n_hits_ratio", ascending=False).head(top_n_slides)
        return agg.reset_index()
