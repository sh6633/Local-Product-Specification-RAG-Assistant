from functools import lru_cache

import numpy as np


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing embedding dependency. Run: uv add sentence-transformers"
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def embed_texts(texts: list[str], *, is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefix = "query: " if is_query else "passage: "
    model_inputs = [prefix + text for text in texts]
    embeddings = _load_model().encode(
        model_inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def cosine_scores(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    if document_embeddings.size == 0:
        return np.empty((0,), dtype=np.float32)
    return document_embeddings @ query_embedding
