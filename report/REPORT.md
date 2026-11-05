# EduAssist LLMOps: API Cost, Latency & Prompt Versioning

**Course:** Machine Learning Operations (LLMOps) AIMLCZG523  
**Student:** `Vishal Anand` | **ID:** `2025AE05692` (`2025AE05692@wilp.bits-pilani.ac.in`)  
**Assignment:** LLMOps Assignment - API Cost, Latency Benchmarking & Prompt Versioning  
**Tracking URI:** `sqlite:///mlflow.db` (`http://127.0.0.1:5000`)  
**Execution Backend:** Local Ollama (`llama3.2:1b`) with OpenAI Python SDK compatibility layer  

---

## 1. Executive Summary & Objective

LLMOps builds upon traditional MLOps principles by managing the lifecycle of applications driven by Large Language Models. While classical MLOps focuses on model training pipelines, dataset drift, feature stores, and weight deployment, LLMOps treats prompts, context retrieval (RAG), guardrails, and non-deterministic text outputs as first-class versioned production artifacts. Operational success in LLMOps centers around controlling API token costs, minimizing autoregressive latency, and enforcing strict output contracts.

In this study, we operationalize **EduAssist AI**, an academic assistant answering undergraduate computer science questions. Using **MLflow 3.17.2** and a local inference backend powered by **Ollama (`llama3.2:1b`)**, we test three prompt engineering strategies across latency, token economics, schema compliance, and promotion management.

| Metric / Parameter | Prompt v1 (Basic) | Prompt v2 (Structured) | Prompt v3 (JSON Contract) |
|---|:---:|:---:|:---:|
| **Prompt Template Purpose** | Unconstrained baseline | Persona & concise framing | Strict JSON schema contract |
| **Average Latency** | 3.463s | 1.999s | **0.651s (-78.2% vs v1)** |
| **Average Input Tokens** | 43.4 | 65.4 | 99.4 |
| **Average Output Tokens** | 448.6 | 363.2 | **112.2 (-75.0% vs v1)** |
| **Total Average Tokens** | 492.0 | 428.6 | **211.6** |
| **Valid JSON Pass Rate** | 0.0% (0/5) | 0.0% (0/5) | **100.0% (5/5)** |
| **Marginal API Token Cost** | $0.00 (Self-hosted) | $0.00 (Self-hosted) | $0.00 (Self-hosted) |
| **Cloud Cost Projection (10k req)** | $2.76 | $2.28 | **$0.82 (-70.3% vs v1)** |
| **MLflow Registry Status** | Version 1 | Version 2 | **Version 3 (@production)** |

---

## 2. Experimental Setup & Questions Corpus

Testing was conducted across 5 foundational computer science questions:
- **Q1:** Difference between supervised and unsupervised learning.
- **Q2:** Explain the bias-variance tradeoff.
- **Q3:** What is overfitting and how can it be prevented?
- **Q4:** What is the purpose of a confusion matrix?
- **Q5:** Explain how gradient descent works.

All evaluations executed with deterministic greedy sampling (`temperature = 0`) using `llama3.2:1b` over the OpenAI Python SDK compatibility interface at `http://localhost:11434/v1`.

---

## 3. Task 1: API Calling & Latency Measurement

For each request, precise start and end wall-clock timestamps were logged along with latency in seconds:

### Task 1 Latency Benchmark Table (Prompt v1)

| Question ID | Model | Start Time | End Time | Latency (s) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | `llama3.2:1b` | 2026-11-03T00:29:03.911 | 2026-11-03T00:29:11.243 | **7.3325** | success |
| **Q2** | `llama3.2:1b` | 2026-11-03T00:29:11.243 | 2026-11-03T00:29:13.975 | 2.7322 | success |
| **Q3** | `llama3.2:1b` | 2026-11-03T00:29:13.976 | 2026-11-03T00:29:16.536 | 2.5604 | success |
| **Q4** | `llama3.2:1b` | 2026-11-03T00:29:16.536 | 2026-11-03T00:29:18.380 | **1.8446** | success |
| **Q5** | `llama3.2:1b` | 2026-11-03T00:29:18.380 | 2026-11-03T00:29:21.223 | 2.8429 | success |

### Latency Summary Statistics (Prompt v1)
- **Mean Latency:** 3.4625 seconds
- **Median Latency:** 2.7322 seconds
- **Minimum Latency:** 1.8446 seconds (Q4)
- **Maximum Latency:** 7.3325 seconds (Q1)
- **Standard Deviation:** 2.1980 seconds

### Interpretation of Latency Variance
The slowest request was **Q1** (7.3325s), while the fastest was **Q4** (1.8446s). The primary factor driving this variation is the autoregressive nature of LLM generation: generation latency is proportional to completion length ($O(N_{out})$). In addition, Q1 was the first request processed by the engine, incurring a runtime initialization and memory caching penalty (cold-start effect). Once the KV-cache and weights were warm in memory, subsequent responses ran steadily around 2.5–2.8 seconds.

---

## 4. Task 2: Token Counting & Cost Economics

Token usage was extracted from the response usage metadata.

### Token Usage and Cost per Request Table (Prompt v1)

| Question ID | Input Tokens | Output Tokens | Total Tokens | Input Cost ($) | Output Cost ($) | Cost per Request ($) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 46 | 422 | 468 | $0.000000 | $0.000000 | $0.000000 |
| **Q2** | 42 | 497 | 539 | $0.000000 | $0.000000 | $0.000000 |
| **Q3** | 46 | 467 | 513 | $0.000000 | $0.000000 | $0.000000 |
| **Q4** | 43 | 337 | 380 | $0.000000 | $0.000000 | $0.000000 |
| **Q5** | 40 | 520 | 560 | $0.000000 | $0.000000 | $0.000000 |
| **Average** | **43.4** | **448.6** | **492.0** | **$0.000000** | **$0.000000** | **$0.000000** |

### Projected Monthly Cost (10,000 Requests)
Under the local Ollama environment, marginal token billing is **$0.00/month**.  
When projecting onto standard commercial cloud APIs (e.g., `gpt-4o-mini` at $0.15/1M input tokens and $0.60/1M output tokens):
$$\text{Input Cost} = \frac{43.4}{10^6} \times 0.15 = \$0.00000651$$
$$\text{Output Cost} = \frac{448.6}{10^6} \times 0.60 = \$0.00026916$$
$$\text{Total Cost per Request} = \$0.00027567$$
$$\text{Projected Monthly Cost (10,000 requests)} = 10,000 \times \$0.00027567 \approx \mathbf{\$2.76\text{/month}}$$

### Cost-Optimization Recommendation
**Enforce concise structured output and cap completion tokens:** In modern LLM billing tiers, output tokens are priced 4x higher than input tokens ($0.60 vs $0.15 per million). In unconstrained prompts like v1, models produce lengthy conversational preambles and redundant summaries (averaging 448.6 completion tokens). Restricting the model to concise bullet points or schema contracts (as in v3) reduces completion length to 112.2 tokens, immediately reducing commercial operating costs from $2.76 to $0.82 per month (a **70.3% recurring cost reduction**).

---

## 5. Task 3: Prompt Versioning & MLflow Evaluation

Three prompt variations were registered in the MLflow Prompt Registry under the model entity `eduassist_qa_prompt`:

- **Prompt v1 (Basic):**  
  `Answer the following student question clearly. Question: {{question}}`
- **Prompt v2 (Structured):**  
  `You are an academic assistant. Answer the following student question in a clear, concise, and accurate manner. Use simple language suitable for undergraduate students. Question: {{question}}`
- **Prompt v3 (JSON Output):**  
  `You are an academic assistant. Answer the student question and return ONLY valid JSON. Do not include markdown, explanations outside JSON, or code fences. Use this exact JSON schema: {"answer": "string", "confidence": "high | medium | low", "key_points": ["string"]}. Question: {{question}}`

### Prompt Comparison Table

| Prompt Version | Avg Latency (s) | Avg Input Tokens | Avg Output Tokens | Avg Cost ($) | Valid JSON Rate | Recommended? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** | 3.463 | 43.4 | 448.6 | 0.000000 | 0.00 (0%) | No |
| **v2** | 1.999 | 65.4 | 363.2 | 0.000000 | 0.00 (0%) | No |
| **v3** | **0.651** | **99.4** | **112.2** | **0.000000** | **1.00 (100%)** | **Yes (@production)** |

---

## 6. Promotion Justification (>150 Words)

Prompt version **v3** is definitively selected and promoted to the `@production` alias in MLflow for the EduAssist application. In production LLMOps environments, frontend user interfaces and automated downstream grading microservices require deterministic contracts rather than unpredictable natural language paragraphs. Prompts v1 and v2 completely failed schema validation with a **0.0% valid JSON rate**, returning unconstrained markdown essays that break API parsing. 

In contrast, Prompt v3 achieved a flawless **100.0% valid JSON pass rate (1.00)**, adhering strictly to the required schema (`answer`, `confidence`, `key_points`). Operationally, v3 delivers outstanding performance gains: average latency drops from **3.463 seconds (v1) to 0.651 seconds (v3)**, representing a **78.2% reduction in end-user wait time**. This acceleration is directly caused by a **75.0% reduction in generated completion tokens** (from 448.6 tokens down to 112.2 tokens). Because autoregressive generation scales sequentially with completion tokens, eliminating conversational fluff substantially minimizes GPU memory bandwidth consumption. Furthermore, under commercial cloud billing, v3 slashes projected monthly inference expenditures by **70.3%** ($0.82 vs $2.76 per 10,000 queries). Prompt v3 provides superior answer precision, deterministic reliability, ultra-low latency, and optimal cost efficiency, making it the clear choice for production deployment.

---

## 7. Deliverables & Repository Layout

```text
files (1)/
├── eduassist_llmops_assignment.ipynb # Full interactive Jupyter notebook with cell outputs
├── run_evaluation.py                 # Automated end-to-end evaluation pipeline
├── requirements.txt                  # Python dependencies (mlflow, openai, pandas, jupyter)
├── README.md                         # Project documentation and setup guide
├── report/
│   ├── REPORT.md                     # Comprehensive LLMOps assignment report
│   ├── REPORT.html                   # HTML formatted report
│   └── EduAssist_LLMOps_Assignment_Report.pdf # Final export PDF
├── latency_tokens_cost.csv           # Detailed granular metrics per question/run
├── prompt_evaluation_results.csv     # Summary benchmark comparison table
├── responses_v1.csv                  # Raw text completions from Prompt v1
├── responses_v2.csv                  # Raw text completions from Prompt v2
├── responses_v3.csv                  # Validated JSON completions from Prompt v3
└── mlflow.db                         # SQLite tracking store for MLflow
```
