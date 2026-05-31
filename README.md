# GIGABYTE AORUS MASTER 16 AM6H RAG 硬體小助手

這是一個針對 **GIGABYTE AORUS MASTER 16 AM6H** 筆電規格建立的本地端 RAG 問答系統。系統可以根據整理後的產品規格資料，回答繁體中文與英文問題，並透過 `llama.cpp` 串流輸出回答。

本專案重點不是使用高階 RAG 框架，而是用純 Python 實作資料整理、檢索、Prompt 組合、模型推論與 benchmark 評測流程。

## 專案限制

- 不使用 LangChain / LlamaIndex
- 使用純 Python 實作 RAG 核心邏輯
- 使用 `uv` 管理 Python 環境
- 使用 `llama.cpp` / `llama-cpp-python` 進行本地推論
- 支援繁體中文與英文混合提問
- 支援 Streaming 串流輸出

## 環境設定與啟動步驟

本專案使用 `uv`：

```powershell
uv sync
```

啟動問答系統：

```powershell
uv run python src\main.py
```

執行 benchmark：

```powershell
uv run python src\run_benchmark_embedding.py
```

## 模型選擇與 4GB VRAM 限制

建議模型：

```text
Qwen2.5-3B-Instruct GGUF Q4_K_M
```

模型放置位置：

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

選擇理由：

- 3B 模型較適合消費級筆電本地推論。
- Q4_K_M 量化可以降低記憶體與 VRAM 需求。
- Qwen 系列對中文與英文混合輸入較穩定。
- GGUF 格式可直接搭配 `llama.cpp` 使用。
- 不依賴雲端 API，適合本地端展示。

若部署環境只有 4GB VRAM，建議使用 CPU 推論：

```python
N_GPU_LAYERS = 0
```

這樣模型權重不會大量佔用 GPU VRAM，可符合 4GB VRAM 限制。若本機 GPU VRAM 較高，也可以開啟部分 GPU offload：

```python
N_GPU_LAYERS = 10
```

Embedding 模型目前固定使用 CPU，避免與 `llama.cpp` 搶 VRAM。

## 專案結構

```text
.
├── data/
│   └── processed/
│       └── chunks_natural.json
├── examples/
│   └── AORUS_MASTER16_AM6H_100Q_Benchmark.txt
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── main.py
│   ├── rag_natural.py
│   └── run_benchmark_embedding.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## 1. 資料解析

本專案將 GIGABYTE AORUS MASTER 16 AM6H 官網規格頁整理成結構化資料，重點是保留規格表中的 **Key-Value 對應關係**。

整理後的主要資料檔案：

```text
data/processed/chunks_natural.json
```

每個 chunk 對應一個規格類別，例如：

```text
CPU
GPU
Display
Memory
Storage
Ports
Connectivity
Webcam
Security
Battery
```

這種整理方式適合規格表資料，因為產品規格不像文章或 PDF 可以直接按照段落切 chunk。此專案改用「規格類別」作為 chunk 單位，例如 GPU chunk 會同時描述 BZH、BYH、BXH 三個 SKU 的 GPU 差異。

除了官網表格化的規格資訊外，本專案也額外加入基於表格內容整理出的 chunks，例如產品摘要、主要差異等。這是為了貼近使用者實際提問習慣，因為使用者不一定會直接問「顯示晶片」或「連接埠」，也可能會問「三款差在哪」、「哪個版本比較強」這類較自然的問題。

此外，每個 chunk 也會加入部分中英文關鍵字，例如 GPU、顯示晶片、Thunderbolt、連接埠、Connectivity、通訊功能等。這是因為檢索流程仍包含 BM25 與詞頻相似度，而這類方法很依賴關鍵字命中，因此適度補充關鍵字可以提升召回率。

## 2. 檢索與生成

RAG Pipeline 位於：

```text
src/rag_natural.py
```

本專案實作方式：

1. **BM25 + embedding rerank**：先用 BM25 召回候選 chunks，再使用 `intfloat/multilingual-e5-small` 建立的 Vector Index 依 embedding similarity 重新排序。

檢索後會將選出的 chunks 組合進 prompt，再交給 `llama.cpp` 進行 Streaming 串流生成。

- `BM25`：處理 RTX 5090、Thunderbolt 5、Wi-Fi 7 等精確關鍵詞。
- `Vector Index / embedding rerank`：改善中英文與語意相近問題，例如「通訊功能」與 networking / connectivity。

生成模型透過 `llama-cpp-python` 呼叫本地 GGUF 模型，並使用 Streaming 逐 token 輸出。

互動式問答啟動方式：

```powershell
uv run python src\main.py
```

範例問題：

```text
BZH、BYH、BXH 主要差在哪？
這台電腦有 RTX 5090 的版本嗎？
請告訴我這台的通訊功能配置為何？
Can I charge it through Thunderbolt?
Which model has the highest VRAM?
```

## 3. 系統評測

Benchmark 題目位於：

```text
examples/AORUS_MASTER16_AM6H_100Q_Benchmark.txt
```

執行 benchmark：

```powershell
uv run python src\run_benchmark_embedding.py
```

只跑前 20 題：

```powershell
uv run python src\run_benchmark_embedding.py --limit 20
```

輸出檔案：

```text
data/processed/benchmark_results_embedding.json
```

每筆結果會記錄：

```json
{
  "question": "...",
  "answer": "...",
  "TTFT": 0.58,
  "TPS": 28.5,
  "token": 91,
  "retrieved_chunks": []
}
```

定量指標：

- **TTFT**：Time To First Token，首字延遲。
- **TPS**：Tokens Per Second，生成速度。
- **token**：回答生成 token 數。

以 `run_benchmark_embedding.py` 對 200 題 benchmark 的結果為例：

| 指標 | 最大值 | 中位數 |
| --- | ---: | ---: |
| TTFT | 1.10 秒 | 0.52 秒 |
| TPS | 32.95 tokens/s | 29.64 tokens/s |
| token 數 | 502 | 24 |

定性分析重點：

- 是否檢索到正確 chunk。
- 回答是否忠於規格資料。
- 是否支援中英文問題。
- 是否能拒答無關問題。
- 是否會對缺失資料產生幻覺。

交叉比對 GPT 與 Claude 對 benchmark 結果的分析後，整理出以下較具代表性的回答問題：

**問題 1：回答過於擴張**

實際問題：

```text
請告訴我這台有哪些 AI 相關功能？
```

模型實際回答：

```text
AORUS MASTER 16 AM6H 提供 AI 相關功能包括 AI Boost、Windows Hello Face Authentication。
AI Boost 可提升 AI 記憶體頻率，並支援 Dynamic Boost 功能。
Windows Hello Face Authentication 是一種臉部辨識技術，用於 Windows Hello 身份驗證。 (｀・ω・´)b
```

應改善之處：

```text
應主要回答 GPU AI Boost：
BZH 1797 MHz、BYH 1902 MHz、BXH 1962 MHz。
Windows Hello 應歸類為安全 / webcam 功能，而不是 AI 功能。
```

**問題 2：檢索到正確資訊但生成端過度保守**

實際問題：

```text
這台電腦有 NVIDIA Advanced Optimus 嗎？
```

模型實際回答：

```text
不好意思喔，這個問題我沒辦法回答喔  (｀・ω・´)b
```

應改善之處：

```text
若 context 明確列出某項功能，模型應直接回答有，
不應在已有證據時拒答。
```

**問題 3：推薦型問題推理不足**

實際問題：

```text
如果我要做 AI 訓練該選哪個版本？
```

模型實際回答：

```text
BYH 版本適合做 AI 訓練。
BYH 使用 NVIDIA GeForce RTX 5080 視覺處理器，
具有 16GB GDDR7 VRAM 和 175W 的最大功耗，適合進行 AI 訓練。 (｀・ω・´)b
```

應改善之處：

```text
若根據 GPU / VRAM 規格推論，AI 訓練應優先考慮 BZH，
因為 BZH 搭載 RTX 5090、24GB VRAM、175W Graphics Power。
```

**問題 4：規格未提及不等於不存在**

實際問題：

```text
Does this laptop have a touchscreen?
```

模型實際回答：

```text
The laptop does not have a touchscreen. (｀・ω・´)b
```

應改善之處：

```text
如果規格沒有提到 touchscreen，
應回答「規格未提及」，而不是斷言「沒有」。
```

**問題 5：中英文一致性與推薦邏輯不穩**

實際問題：

```text
If I want to play AAA games, which version should I choose?
```

模型實際回答：

```text
The context provided does not mention anything about the performance or specifications of AAA games.
It only talks about the ports, keyboard, and storage of the AORUS MASTER 16 AM6H.
Therefore, the provided specifications do not mention which version you should choose for AAA games. (｀・ω・´)b
```

應改善之處：

```text
若允許根據 GPU 規格推論，應推薦 BZH。
若不做推論，也應讓中英文問題採取一致的回答策略。
```

正確率分析(By Claude)：

```text
總題數：200
答對：163
答錯：37
整體正確率：81.5%
```

分類正確率：

| 分類 | 題數 | 答對 | 正確率 | 主要錯誤類型 |
| --- | ---: | ---: | ---: | --- |
| 中文規格查詢 Q1-Q20 | 20 | 19 | 95% | AI Boost 功能描述錯誤 |
| 中文是非題 Q21-Q50 | 20 | 14 | 70% | 有 context 卻拒答、格式異常 |
| 中文使用場景 Q51-Q80 | 20 | 4 | 20% | GPU 排序錯誤、場景建議邏輯不足、大量拒答 |
| 中文超出範圍 Q81-Q100 | 20 | 20 | 100% | 無 |
| 英文規格查詢 Q101-Q120 | 20 | 20 | 100% | 無 |
| 英文是非題 Q121-Q150 | 20 | 16 | 80% | 規格未提及卻斷言沒有、AI PC 標籤混淆 |
| 英文使用場景 Q151-Q180 | 20 | 11 | 55% | GPU 排序錯誤、場景建議邏輯不足 |
| 英文超出範圍 Q181-Q200 | 20 | 19 | 95% | 超出範圍題的拒答格式仍可更一致 |


觀察以及結論:

- prompt 對模型的約束力很大，但也不穩定。
- 模型能力會影響上下文長度，但上下文長度也會影響模型能力。
- 模型對一些關鍵字及其敏感，但不一定根據該事實做有效推理。
- RAG還有很多改善空間。
- 即使檢索內容正確，開放式問題仍仰賴模型能力。
