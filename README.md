# Cross-Sell Engine

## Required Components

- PostgreSQL 18
  - Database Name: "cross-sales-engine-dev-db1"
- a File Storage
- Python 3.11
- Ollama
  - Embedding Model: mxbai-embed-large
  - LLM: llama3.1

## To Start

**Do Once**

> python3 -m venv .venv

> . ./.venv/bin/activate

> pip install -r requirements.txt

Set Environment Variable:

> export AZURE_API_KEY="..."

**When Starting**

> . ./.venv/bin/activate
> python3 app.py

**When Stopping**

> ps aux
>
> kill PID

## Available Endpoints

- /inventory
- /chat
- /chats/TEST_CHAT
- /customers
