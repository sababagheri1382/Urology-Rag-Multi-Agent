from fastapi import FastAPI
from app.api.openai_compatible import router

app = FastAPI(
    title="Urology RAG API",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Urology RAG API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }