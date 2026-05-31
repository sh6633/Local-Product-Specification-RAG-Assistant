import json
import math
import re
from collections import Counter
from pathlib import Path

from config import NATURAL_CHUNKS_PATH, TOP_K
from embeddings import cosine_scores, embed_texts


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
EMBEDDING_CANDIDATE_COUNT = 8


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def response_language_instruction(question: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", question))
    if has_cjk:
        return "The user question language is Traditional Chinese. Answer in Traditional Chinese."
    return "The user question language is English. Answer in English."


def load_chunks(path: Path = NATURAL_CHUNKS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class NaturalRAGPipeline:
    def __init__(self, chunks_path: Path = NATURAL_CHUNKS_PATH, top_k: int = TOP_K) -> None:
        self.chunks = load_chunks(chunks_path)
        self.top_k = top_k
        self.chunk_tokens = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.doc_freq = self._build_doc_freq()
        self.avg_doc_len = self._average_doc_length()
        self.chunk_embeddings = embed_texts(
            [chunk["text"] for chunk in self.chunks],
            is_query=False,
        )

    def _build_doc_freq(self) -> Counter:
        doc_freq = Counter()
        for tokens in self.chunk_tokens:
            doc_freq.update(set(tokens))
        return doc_freq

    def _average_doc_length(self) -> float:
        if not self.chunk_tokens:
            return 0.0
        return sum(len(tokens) for tokens in self.chunk_tokens) / len(self.chunk_tokens)

    def _bm25_score(self, query_tokens: list[str], chunk_index: int) -> float:
        tokens = self.chunk_tokens[chunk_index]
        if not tokens:
            return 0.0

        term_counts = Counter(tokens)
        doc_count = len(self.chunk_tokens)
        doc_len = len(tokens)
        k1 = 1.5
        b = 0.75
        score = 0.0

        for token in query_tokens:
            tf = term_counts[token]
            if tf == 0:
                continue
            df = self.doc_freq[token]
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1))
            score += idf * (tf * (k1 + 1)) / denominator

        return score

    def _similarity_score(self, query_tokens: list[str], chunk_index: int) -> float:
        query_counts = Counter(query_tokens)
        chunk_counts = Counter(self.chunk_tokens[chunk_index])
        if not query_counts or not chunk_counts:
            return 0.0

        shared = set(query_counts) & set(chunk_counts)
        dot = sum(query_counts[token] * chunk_counts[token] for token in shared)
        query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
        chunk_norm = math.sqrt(sum(value * value for value in chunk_counts.values()))
        if query_norm == 0 or chunk_norm == 0:
            return 0.0
        return dot / (query_norm * chunk_norm)

    def retrieve(self, question: str) -> list[dict]:
        query_tokens = tokenize(question)
        scored = []

        for index, chunk in enumerate(self.chunks):
            bm25 = self._bm25_score(query_tokens, index)
            similarity = self._similarity_score(query_tokens, index)
            if bm25 == 0:
                continue
            weighted_similarity = similarity * 10
            score = bm25 + weighted_similarity
            scored.append(
                {
                    "chunk": chunk,
                    "chunk_index": index,
                    "score": score,
                    "bm25": bm25,
                    "lexical_similarity": similarity,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        if not scored:
            return []

        # Embedding rerank added here:
        # 1. BM25 + token-count cosine first recalls a small candidate set.
        # 2. multilingual-e5-small embeds the question and candidate chunks.
        # 3. Final top-k chunks are selected by real embedding cosine similarity.
        candidates = scored[: max(self.top_k, EMBEDDING_CANDIDATE_COUNT)]
        query_embedding = embed_texts([question], is_query=True)[0]
        candidate_embeddings = self.chunk_embeddings[
            [item["chunk_index"] for item in candidates]
        ]
        embedding_scores = cosine_scores(query_embedding, candidate_embeddings)

        for item, embedding_score in zip(candidates, embedding_scores, strict=True):
            item["embedding_similarity"] = float(embedding_score)

        candidates.sort(key=lambda item: item["embedding_similarity"], reverse=True)

        return [
            {
                **item["chunk"],
                "_score": round(item["embedding_similarity"], 4),
                "_bm25": round(item["bm25"], 4),
                "_similarity": round(item["embedding_similarity"], 4),
                "_lexical_similarity": round(item["lexical_similarity"], 4),
            }
            for item in candidates[: self.top_k]
        ]

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[{index}]\n{chunk['text']}"
            for index, chunk in enumerate(chunks, start=1)
        )
        language_instruction = response_language_instruction(question)

        return f"""Context:
{context}

Question:
{question}

Important:
{language_instruction}
Use only the context above.
If the question is unrelated to the product, answer exactly:
\u4e0d\u597d\u610f\u601d\u5594\uff0c\u9019\u500b\u554f\u984c\u6211\u6c92\u8fa6\u6cd5\u56de\u7b54\u5594  (\uff40\u30fb\u03c9\u30fb\u00b4)b

Answer:"""
