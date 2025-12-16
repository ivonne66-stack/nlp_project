# Quickstart (macOS, Python 3.11 recommended)

## Prepare the environment

1. Install Python 3.11 (Homebrew: `brew install python@3.11`).
2. Create and activate a virtual environment in the repo root:
   ```bash
   python3.11 -m venv .venv311 && source .venv311/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
   The first run downloads all-MiniLM-L6-v2 from HuggingFace, so you need network access.

## Configure .env (place in repo root)

```
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
RECIPE_DIR=./data/pan-pal-recipe-embeddings/outputs
SUB_DIR=./data/pan-pal-ingredient-substitutions/outputs
```

Ensure the `data` paths above already contain the embeddings/JSON files.

## Start the backend (activate venv first)

```bash
python3 -m dotenv -f .env run -- uvicorn backend.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

## Access frontend/API

Frontend: open http://127.0.0.1:8000/ (or `/web`).
