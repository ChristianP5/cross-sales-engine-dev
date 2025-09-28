
# Cross-Sell Engine

_Required Components_
- PostgreSQL 18
    - Database Name: "cross-sales-engine-dev-db1"
- a File Storage
- Python 3.11
- Ollama
    - Embedding Model: mxbai-embed-large
    - LLM: llama3

_To Start_

**Do Once**
> python3 -m venv .venv

> . ./.venv/bin/activate

> pip install -r requirements.txt

**When Starting**
> python3 app.py
