# VM Setup & Execution Guide

This guide provides the exact step-by-step commands to run the EduAssist LLMOps pipeline inside a Virtual Machine (cloud VM, headless server, or local hypervisor like Hyper-V, VirtualBox, or WSL2) right after running `git clone`.

---

## 1. Linux Virtual Machine (Ubuntu / Debian)

### Step 1: Navigate into the cloned repository
```bash
cd mlops2-assign
```

### Step 2: Install Ollama and pull the model
```bash
# 1. Install Ollama (official one-line installer)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the 1B parameter lightweight model
ollama pull llama3.2:1b
```
*(On Linux, the Ollama service starts automatically as a systemd background service on port 11434).*

### Step 3: Create and activate Python virtual environment
```bash
# Ensure Python venv and pip are installed
sudo apt update && sudo apt install -y python3-venv python3-pip

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install assignment dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the evaluation script
```bash
python run_evaluation.py
```
This executes the experiment across prompts `v1`, `v2`, and `v3`, generates the CSV files (`latency_tokens_cost.csv`, `prompt_evaluation_results.csv`, `responses_v*.csv`), logs metrics to `mlflow.db`, and sets the `@production` alias on `v3`.

### Step 5: Start the MLflow UI for Remote Access
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
> **Tip:** By binding to `--host 0.0.0.0`, you can access the dashboard directly from your host machine's browser at:
> `http://<VM-IP-ADDRESS>:5000`
> This makes taking the required MLflow screenshots very convenient.

---

## 2. Windows Virtual Machine

### Step 1: Navigate into the repository
```powershell
cd mlops2-assign
```

### Step 2: Install Ollama & pull model
```powershell
# If Ollama is not installed yet:
winget install Ollama.Ollama

# Pull the model
ollama pull llama3.2:1b
```

### Step 3: Create and activate virtual environment
```powershell
# Set execution policy if script activation is restricted
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run evaluation pipeline
```powershell
python run_evaluation.py
```

### Step 5: Start MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Open `http://localhost:5000` (or `http://<VM-IP-ADDRESS>:5000` from the host) in your browser to view your runs and take the screenshots.

