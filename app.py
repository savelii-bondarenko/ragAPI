import uvicorn
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Python RAG Worker")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)