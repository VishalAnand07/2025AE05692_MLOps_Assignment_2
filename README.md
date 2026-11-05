# EduAssist LLMOps Assignment

## Environment
- Python 3.10+ (record your exact version: `python --version`)
- Libraries: see `requirements.txt` (`mlflow>=3.1`, `openai>=1.40`, `pandas>=2.0`, `jupyter`)
- Environment variables:
  - **Option A (Local Ollama - Free / Offline)**:
    - Install Ollama: `winget install Ollama.Ollama` or download from https://ollama.com/download
    - Pull model: `ollama run llama3.2:1b` (or `llama3.2`)
    - Set env variables (optional / defaults already built into the notebook):
      - Windows PowerShell:
        ```powershell
        $env:USE_LOCAL_OLLAMA="true"
        $env:MODEL="llama3.2:1b"
        $env:OPENAI_BASE_URL="http://localhost:11434/v1"
        $env:OPENAI_API_KEY="ollama"
        ```
  - **Option B (OpenAI Cloud)**:
    - `OPENAI_API_KEY` (required)
    - `USE_LOCAL_OLLAMA="false"`
    - `MODEL="gpt-4o-mini"`

## Install
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce
1. Start Ollama: verify `http://localhost:11434` is running (`ollama serve` or Ollama system tray).
2. Launch notebook: `jupyter notebook eduassist_llmops_assignment.ipynb`
3. Verify settings in the Config cell (local Ollama defaults to `llama3.2:1b` with `$0.00` marginal token cost).
4. Run all cells top to bottom. This creates `latency_tokens_cost.csv`, `prompt_evaluation_results.csv`, `mlflow.db`.
5. Start MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` -> http://127.0.0.1:5000
6. Screenshot: experiment runs + metrics, Prompt Registry (v1-v3), prompt compare/diff, `production` alias.
7. Write the report (500-700 words) and export notebook to PDF/HTML.
