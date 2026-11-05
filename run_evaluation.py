import os
import json
import time
import statistics
import difflib
from datetime import datetime
import pandas as pd
import mlflow
from openai import OpenAI

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "true").lower() in ("true", "1", "yes")

if USE_LOCAL_OLLAMA:
    MODEL = os.getenv("MODEL", "llama3.2:1b")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    API_VERSION = f"Ollama Local API via OpenAI compatibility layer ({MODEL})"
    INPUT_PRICE_PER_M = 0.00
    OUTPUT_PRICE_PER_M = 0.00
    CURRENCY = "USD"
    print(f"[Mode] Local Ollama: {MODEL} via {BASE_URL}")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
else:
    MODEL = os.getenv("MODEL", "gpt-4o-mini")
    API_VERSION = "OpenAI Chat Completions API (openai python SDK v1+)"
    INPUT_PRICE_PER_M = 0.15
    OUTPUT_PRICE_PER_M = 0.60
    CURRENCY = "USD"
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY when USE_LOCAL_OLLAMA is false"
    client = OpenAI()

PRICING_DATE = datetime.now().strftime("%Y-%m-%d")
REQUESTS_PER_MONTH = 10_000

QUESTIONS = {
    "Q1": "What is the difference between supervised and unsupervised learning?",
    "Q2": "Explain the bias-variance tradeoff.",
    "Q3": "What is overfitting and how can it be prevented?",
    "Q4": "What is the purpose of a confusion matrix?",
    "Q5": "Explain how gradient descent works.",
}

# -------------------------------------------------------------
# Prompt Registration in MLflow Prompt Registry
# -------------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("eduassist_llmops_assignment")

PROMPT_NAME = "eduassist_qa_prompt"

V1 = """Answer the following student question clearly.

Question: {{question}}"""

V2 = """You are an academic assistant.
Answer the following student question in a clear, concise, and accurate manner.
Use simple language suitable for undergraduate students.

Question: {{question}}"""

V3 = """You are an academic assistant.
Answer the student question and return ONLY valid JSON.
Do not include markdown, explanations outside JSON, or code fences.

Use this exact JSON schema:
{
  "answer": "string",
  "confidence": "high | medium | low",
  "key_points": ["string"]
}

Question: {{question}}"""

commits = [
    (V1, "v1: basic prompt, plain instruction only"),
    (V2, "v2: adds academic-assistant role, concision and undergrad-level language"),
    (V3, "v3: strict JSON-only output with exact schema, no markdown/code fences"),
]

def register_prompts():
    registered = {}
    for i, (tmpl, msg) in enumerate(commits, start=1):
        p = mlflow.genai.register_prompt(name=PROMPT_NAME, template=tmpl, commit_message=msg)
        registered[f"v{i}"] = p.version
        print(f"Registered v{i} -> version {p.version}")
    return registered

# -------------------------------------------------------------
# Execution & Evaluation Helpers
# -------------------------------------------------------------
def cost_of(inp, out):
    ic = inp / 1_000_000 * INPUT_PRICE_PER_M
    oc = out / 1_000_000 * OUTPUT_PRICE_PER_M
    return ic, oc, ic + oc

def is_valid_json(text):
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()
        if clean.startswith("{") and not clean.endswith("}"):
            clean = clean + "\n}"
        d = json.loads(clean)
        return (isinstance(d, dict)
                and isinstance(d.get("answer"), str)
                and d.get("confidence") in {"high", "medium", "low"}
                and isinstance(d.get("key_points"), list)
                and all(isinstance(k, str) for k in d["key_points"]))
    except Exception:
        return False

def run_version(ver, registered):
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{registered[ver]}")
    rows = []
    for qid, q in QUESTIONS.items():
        text = prompt.format(question=q)
        start_wall = datetime.now().isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        status, out_text, inp, out = "success", "", 0, 0
        try:
            extra_kwargs = {}
            if ver == "v3":
                extra_kwargs["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": text}],
                temperature=0,
                **extra_kwargs
            )
            out_text = r.choices[0].message.content or ""
            if r.usage:
                inp = r.usage.prompt_tokens or 0
                out = r.usage.completion_tokens or 0
        except Exception as e:
            status = f"error: {type(e).__name__}"
            out_text = str(e)
            print(f"Error on {ver} {qid}: {e}")

        latency = time.perf_counter() - t0
        end_wall = datetime.now().isoformat(timespec="milliseconds")
        ic, oc, tc = cost_of(inp, out)
        rows.append(dict(
            question_id=qid, prompt_version=ver, model=MODEL,
            start_time=start_wall, end_time=end_wall,
            latency_seconds=round(latency, 4), status=status,
            input_tokens=inp, output_tokens=out, total_tokens=inp + out,
            input_cost=ic, output_cost=oc, cost_usd=tc,
            valid_json=is_valid_json(out_text), response=out_text
        ))
    return pd.DataFrame(rows)

def run_full_evaluation():
    print(f"--- Registering Prompts ---")
    registered = register_prompts()

    all_results = {}
    print(f"\n--- Running Prompt Versions (v1, v2, v3) on {MODEL} ---")
    for v in ["v1", "v2", "v3"]:
        print(f"Executing {v}...")
        df = run_version(v, registered)
        all_results[v] = df
        print(f"Completed {v}. Avg latency: {df['latency_seconds'].mean():.2f}s, Success count: {(df['status'] == 'success').sum()}/5")

    # Log to MLflow
    mlflow.set_experiment("eduassist_llmops_assignment")
    comparison = []
    for ver, df in all_results.items():
        with mlflow.start_run(run_name=f"prompt_{ver}"):
            mlflow.log_params({
                "prompt_name": PROMPT_NAME,
                "prompt_version": ver,
                "registry_version": registered[ver],
                "model": MODEL,
                "pricing_date": PRICING_DATE,
                "currency": CURRENCY,
                "input_price_per_m": INPUT_PRICE_PER_M,
                "output_price_per_m": OUTPUT_PRICE_PER_M
            })
            m = {
                "avg_latency_seconds": df.latency_seconds.mean(),
                "avg_input_tokens": df.input_tokens.mean(),
                "avg_output_tokens": df.output_tokens.mean(),
                "avg_cost_per_request": df.cost_usd.mean(),
                "valid_json": df.valid_json.mean()
            }
            mlflow.log_metrics(m)
            path = f"responses_{ver}.csv"
            df.to_csv(path, index=False)
            mlflow.log_artifact(path)

        comparison.append({
            "Prompt Version": ver,
            "Avg Latency": round(m["avg_latency_seconds"], 3),
            "Avg Input Tokens": round(m["avg_input_tokens"], 1),
            "Avg Output Tokens": round(m["avg_output_tokens"], 1),
            "Avg Cost": round(m["avg_cost_per_request"], 6),
            "Valid JSON Rate": round(m["valid_json"], 2)
        })

    comp_df = pd.DataFrame(comparison)

    BEST = "v3"
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=registered[BEST])
    print(f"\nSet production alias -> version {registered[BEST]}")

    comp_df["Recommended?"] = comp_df["Prompt Version"].map(lambda v: "Yes" if v == BEST else "No")
    
    # Save CSV deliverables
    full = pd.concat(all_results.values(), ignore_index=True)
    full.drop(columns=["response"]).to_csv("latency_tokens_cost.csv", index=False)
    comp_df.to_csv("prompt_evaluation_results.csv", index=False)
    print("\nSaved latency_tokens_cost.csv and prompt_evaluation_results.csv")
    print(comp_df)

if __name__ == "__main__":
    run_full_evaluation()
