# app/main.py  -- run from the project root: uvicorn app.main:app --reload
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import generate_sql, load_everything

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app):
    load_everything()          # load the model once at startup, not on the first request
    yield


app = FastAPI(title="Text-to-SQL", lifespan=lifespan)


class Request(BaseModel):
    question: str = Field(max_length=500)
    columns: str = Field(max_length=1000)


@app.post("/api/translate")
def translate(req: Request):          # plain `def`: FastAPI runs it in a thread pool, so
    return generate_sql(req.question, req.columns)   # CPU-heavy decoding doesn't block the server


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")