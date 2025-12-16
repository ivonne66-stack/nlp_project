from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import faiss  # type: ignore
import numpy as np
import google.generativeai as genai  # type: ignore
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer  # type: ignore
from starlette.responses import FileResponse, RedirectResponse
from dotenv import load_dotenv
from pydantic import BaseModel

"""
Port of the Kaggle notebook demo to a FastAPI backend.
Uses local data by default (data/pan-pal-*/outputs) and optionally Gemini.
"""

load_dotenv()

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RECIPE_DIR = Path(
    os.getenv(
        "RECIPE_DIR",
        DEFAULT_DATA_ROOT / "pan-pal-recipe-embeddings" / "outputs",
    )
).expanduser()
SUB_DIR = Path(
    os.getenv(
        "SUB_DIR",
        DEFAULT_DATA_ROOT / "pan-pal-ingredient-substitutions" / "outputs",
    )
).expanduser()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_recipe_data():
    if not RECIPE_DIR.exists():
        raise RuntimeError(f"RECIPE_DIR not found: {RECIPE_DIR}")

    recipe_embs = np.load(RECIPE_DIR / "recipe_embeddings.npy")
    index = faiss.read_index(str(RECIPE_DIR / "faiss_index.bin"))
    metadata = json.load(open(RECIPE_DIR / "recipe_metadata.json"))

    recipe_files = [
        f
        for f in os.listdir(RECIPE_DIR)
        if "recipe" in f and f.endswith(".json") and "metadata" not in f
    ]
    if recipe_files:
        recipes = json.load(open(RECIPE_DIR / recipe_files[0]))
    else:
        recipes = [
            {
                "id": i,
                "ingredients_raw": m.get("ingredients", []),
                "directions": m.get("directions", []),
            }
            for i, m in enumerate(metadata)
        ]
    return recipe_embs, index, metadata, recipes


def load_substitution_data():
    if not SUB_DIR.exists():
        raise RuntimeError(f"SUB_DIR not found: {SUB_DIR}")
    return json.load(open(SUB_DIR / "ingredient_substitution_final.json"))


# ---------------------------------------------------------------------------
# Recipe database
# ---------------------------------------------------------------------------

class RecipeDB:
    def __init__(self, encoder, index, metadata, recipes, sub_data):
        self.encoder = encoder
        self.index = index
        self.metadata = metadata
        self.recipes = recipes
        self.sub_data = sub_data
        self.recipe_by_id = {r["id"]: r for r in recipes}

    def search(self, query: str, top_k: int = 5):
        top_k = int(top_k)
        query_vec = self.encoder.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            recipe = self.recipes[meta["id"]]
            results.append(
                {
                    "id": meta["id"],
                    "title": meta["title"],
                    "score": float(dist),
                    "ingredients": recipe.get("ingredients_raw", []),
                    "directions": recipe.get("directions", []),
                }
            )
        return results

    def get_substitutes(self, ingredient: str):
        ing = ingredient.lower().strip()
        if ing in self.sub_data:
            subs = self.sub_data[ing]["substitutes"]
            return {
                "ingredient": ing,
                "substitutes": [
                    {
                        "name": s["name"],
                        "ratio": s["ratio"],
                        "context": s.get("context", "general"),
                        "source": s.get("source", "N/A"),
                    }
                    for s in subs[:5]
                ],
            }
        for key in self.sub_data.keys():
            if ing in key or key in ing:
                subs = self.sub_data[key]["substitutes"]
                return {
                    "ingredient": key,
                    "substitutes": [
                        {
                            "name": s["name"],
                            "ratio": s["ratio"],
                            "context": s.get("context", "general"),
                            "source": s.get("source", "N/A"),
                        }
                        for s in subs[:5]
                    ],
                }
        raise HTTPException(status_code=404, detail=f"No substitutes for {ingredient}")

    def find_similar(self, title: str, top_k: int = 5):
        top_k = int(top_k)
        idx = None
        for i, meta in enumerate(self.metadata):
            if title.lower() in meta["title"].lower():
                idx = i
                break
        if idx is None:
            raise HTTPException(status_code=404, detail="Recipe not found")

        query = recipe_embs[idx : idx + 1].astype("float32")
        _, indices = self.index.search(query, top_k + 1)

        similar = []
        for i in indices[0]:
            if i != idx and i < len(self.metadata):
                meta = self.metadata[i]
                similar.append({"title": meta["title"], "id": meta["id"]})
        base_meta = self.metadata[idx]
        base_recipe = self.recipes[base_meta["id"]]
        return {
            "base": {
                "id": base_meta["id"],
                "title": base_meta["title"],
                "ingredients": base_recipe.get("ingredients_raw", []),
                "directions": base_recipe.get("directions", []),
            },
            "similar": similar[:top_k],
        }

    def get_details_by_title(self, title: str):
        for meta in self.metadata:
            if title.lower() in meta["title"].lower():
                return self.get_details_by_id(meta["id"])
        raise HTTPException(status_code=404, detail="Recipe not found")

    def get_details_by_id(self, recipe_id: int):
        if recipe_id not in self.recipe_by_id:
            raise HTTPException(status_code=404, detail="Recipe not found")
        r = self.recipe_by_id[recipe_id]
        meta_title = next(
            (m["title"] for m in self.metadata if m["id"] == recipe_id), ""
        )
        return {
            "id": recipe_id,
            "title": meta_title,
            "ingredients": r.get("ingredients_raw", []),
            "directions": r.get("directions", []),
        }


# ---------------------------------------------------------------------------
# Gemini tool wrappers
# ---------------------------------------------------------------------------

def search_recipes(query: str, top_k: int = 5):
    return require_db().search(query, top_k)


def get_ingredient_substitutes(ingredient: str):
    return require_db().get_substitutes(ingredient)


def get_recipe_details(recipe_title: str):
    return require_db().get_details_by_title(recipe_title)


def find_similar_recipes(recipe_title: str, top_k: int = 5):
    return require_db().find_similar(recipe_title, top_k)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PanPal Demo API", version="0.2.0")

recipe_embs = None
db: RecipeDB | None = None
load_error: Optional[str] = None
gemini_model = None


@app.get("/health")
def health():
    return {"status": "ok" if db else "error", "detail": load_error}


def require_db():
    if not db:
        raise HTTPException(status_code=500, detail=load_error or "Data not loaded")
    return db


try:
    recipe_embs, faiss_index, metadata, recipes = load_recipe_data()
    substitutions = load_substitution_data()
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    db = RecipeDB(encoder, faiss_index, metadata, recipes, substitutions)
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=[
                search_recipes,
                get_ingredient_substitutes,
                get_recipe_details,
                find_similar_recipes,
            ],
            system_instruction="""You are a proactive chef assistant. 

KEY BEHAVIORS:
1. When user says "I want to make X", IMMEDIATELY call search_recipes(X) to show options.
2. When discussing substitutes, ALWAYS include them in your response.
3. Keep track of recipes you mentioned - when user says "the first one", you know which recipe.
4. Be specific and actionable - don't ask what they want if they already told you.
5. When responding with substitutes, list each substitute as a separate bullet or line. Do not combine them into a single sentence. After listing the substitutes, always ask the user which one they would like to use.
6. When a user asks for the next step, never stop at a single step—always provide all remaining steps.
7. After the user selects a dish, always ask what quantity they want to make. Then adjust and return the recipe based on that quantity. Ask for the quantity first. Once the user responds, generate the recipe adjusted to that amount. Avoid highly precise decimal quantities. Use reasonable, real-world approximations that are practical for cooking. If an ingredient quantity is a decimal, keep only one digit after the decimal point.Very important! 
8. VERY IMPORTANT: Only ask the user how much they want to make, never mention the original recipe size, and always scale all ingredient amounts proportionally—for example, if the requested quantity doubles, every ingredient amount must also double.
9. VERY IMPORTANT: whenever the user asks for a "recipe" or "full recipe",
   you MUST return BOTH:
   - a list of ingredients, AND
   - a clear, step-by-step list of directions.
   Use get_recipe_details(...) or the directions returned by search_recipes(...) to get the steps.
   Never stop after ingredients only.

Example good flow:
User: "I want to make cookies"
You: [Call search_recipes("cookies")] Then show the results immediately

User: "I don't have eggs"  
You: [Call get_ingredient_substitutes("eggs")] Then explain the substitutes

User: "Show me the first one"
You: [Call get_recipe_details with the first recipe title you mentioned earlier]
     Then show ingredients AND step-by-step directions.
""",
        )
except Exception as exc:  # startup failure shows in /health
    load_error = str(exc)


CHATS: dict[str, object] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Minimal chat endpoint: behaves like the Kaggle notebook helper.
    Body example: {"message": "I want to make cookies", "session_id": "demo"}
    """
    session_id = req.session_id or "default"

    # Short-circuit echo for debugging to ensure the endpoint responds
    if os.getenv("CHAT_ECHO_ONLY", "0") == "1":
        return {
            "session_id": session_id,
            "reply": f"(echo) {req.message}",
            "gemini_ready": bool(gemini_model),
        }

    if not gemini_model:
        # Return a friendly message instead of empty reply
        return {
            "session_id": session_id,
            "reply": "Gemini is not configured on the server. Set GEMINI_API_KEY and restart.",
        }
    chat_obj = CHATS.get(session_id)
    if chat_obj is None:
        chat_obj = gemini_model.start_chat(enable_automatic_function_calling=True)
        CHATS[session_id] = chat_obj

    try:
        resp = chat_obj.send_message(req.message)
        return {"session_id": session_id, "reply": resp.text}
    except Exception as exc:  # return a clear error instead of empty reply
        raise HTTPException(status_code=500, detail=f"Gemini error: {exc}")


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


@app.get("/")
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return RedirectResponse(url="/docs")


if FRONTEND_DIR.exists():
    app.mount("/web", StaticFiles(directory=FRONTEND_DIR, html=True), name="web")
