import time
from collections.abc import Iterator

from config import MAX_TOKENS, MODEL_PATH, N_CTX, N_GPU_LAYERS, TEMPERATURE


SYSTEM_PROMPT = """You are a precise product specification assistant for GIGABYTE AORUS MASTER 16 AM6H.

Product naming:
- AORUS MASTER 16 is the product series.
- AM6H is the model.
- BZH, BYH, and BXH are SKUs / configuration variants under this model.

Rules:
- Use only the provided context to answer product specification questions.
- If the question is unrelated to the product, answer exactly: \u4e0d\u597d\u610f\u601d\u5594\uff0c\u9019\u500b\u554f\u984c\u6211\u6c92\u8fa6\u6cd5\u56de\u7b54\u5594  (\uff40\u30fb\u03c9\u30fb\u00b4)b
- If the context does not contain enough information to answer, say that the provided specifications do not mention it.
- If a feature, port, technology, certification, SKU, or model name appears explicitly in the context, do not answer that it is absent.
- If numeric specifications are present in the context, quote them exactly.
- Answer in the same language as the user's question.
- End every answer with this exact emoticon: (\uff40\u30fb\u03c9\u30fb\u00b4)b

Examples:
Q: Please tell me the color of this laptop.
A: The color of this laptop is Dark Tide. (\uff40\u30fb\u03c9\u30fb\u00b4)b
"""

# Previous examples kept for prompt experiments:
# Q: \u8acb\u544a\u8a34\u6211\u9019\u53f0\u7684\u8a18\u61b6\u9ad4\u914d\u7f6e\u70ba\u4f55\uff1f
# A: \u9019\u53f0\u652f\u63f4\u6700\u9ad8 64GB DDR5 5600MHz \u8a18\u61b6\u9ad4\uff0c\u4e26\u63d0\u4f9b 2 \u500b SO-DIMM \u63d2\u69fd\u3002 (\uff40\u30fb\u03c9\u30fb\u00b4)b
#
# Q: \u4eca\u5929\u53f0\u5317\u7684\u5929\u6c23\u5982\u4f55\uff1f
# A: \u4e0d\u597d\u610f\u601d\u5594\uff0c\u9019\u500b\u554f\u984c\u6211\u6c92\u8fa6\u6cd5\u56de\u7b54\u5594  (\uff40\u30fb\u03c9\u30fb\u00b4)b
#
# Q: What is blockchain?
# A: \u4e0d\u597d\u610f\u601d\u5594\uff0c\u9019\u500b\u554f\u984c\u6211\u6c92\u8fa6\u6cd5\u56de\u7b54\u5594  (\uff40\u30fb\u03c9\u30fb\u00b4)b


class LocalLLM:
    def __init__(self) -> None:
        from llama_cpp import Llama

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )

    def stream(self, prompt: str) -> Iterator[tuple[str, dict | None]]:
        started_at = time.perf_counter()
        first_token_at = None
        token_count = 0

        stream = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature= TEMPERATURE,
            stream=True,
        )

        for event in stream:
            delta = event["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if not token:
                continue

            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_count += 1
            yield token, None

        ended_at = time.perf_counter()
        ttft = (first_token_at - started_at) if first_token_at else None
        decode_time = ended_at - (first_token_at or started_at)
        tps = token_count / decode_time if decode_time > 0 else 0.0
        yield "", {"ttft": ttft, "tps": tps, "tokens": token_count}
