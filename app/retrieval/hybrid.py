from app.models.domain import Chunk


def reciprocal_rank_fusion(
    vector_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    top_k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> list[tuple[Chunk, float]]:
    """
    Combines dense and sparse search rankings using weighted Reciprocal Rank Fusion (RRF).
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    # Dense rankings
    for rank, (chunk, score) in enumerate(vector_results):
        chunk_map[chunk.id] = chunk
        rrf_score = dense_weight * (1.0 / (rrf_k + rank + 1))
        scores[chunk.id] = scores.get(chunk.id, 0.0) + rrf_score

    # BM25 rankings
    for rank, (chunk, score) in enumerate(bm25_results):
        chunk_map[chunk.id] = chunk
        rrf_score = bm25_weight * (1.0 / (rrf_k + rank + 1))
        scores[chunk.id] = scores.get(chunk.id, 0.0) + rrf_score

    if not scores:
        return []

    # Sort descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_rrf = sorted_items[0][1] if sorted_items else 1.0

    output: list[tuple[Chunk, float]] = []
    for chunk_id, rrf_score in sorted_items[:top_k]:
        # Normalize score into [0.0, 1.0]
        norm_score = min(1.0, float(rrf_score / max_rrf))
        output.append((chunk_map[chunk_id], norm_score))

    return output
