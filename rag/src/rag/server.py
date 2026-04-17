import os
import tempfile
import asyncio
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, ConfigDict

from rag import prepare_rag_assets, RAGGraph

app = FastAPI(title="RAG Microservice API")

active_indexes: dict[str, RAGGraph] = {}


class QueryRequest(BaseModel):
    index_id: str
    message: str
    message_history: list[str] = []

    model_config = ConfigDict(extra='forbid')


def generate_id() -> str:
    return str(uuid.uuid4())


def get_answer_sync(rag_instance: RAGGraph, msg: str) -> str:
    return rag_instance.get_query(msg)["text"]


def initialize_rag_sync(path: str) -> RAGGraph:
    chunks, embedder, vector_db = prepare_rag_assets(path)
    return RAGGraph(chunks, embedder, vector_db)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    index_id = generate_id()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        new_rag_instance: RAGGraph = await asyncio.to_thread(initialize_rag_sync, tmp_path)
        active_indexes[index_id] = new_rag_instance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return {
        "status": "File processed and RAG initialized",
        "index_id": index_id
    }


@app.post("/query")
async def query_rag(request: QueryRequest):
    if request.index_id not in active_indexes:
        raise HTTPException(status_code=404, detail="Index not found. Please upload the file again.")

    current_rag: RAGGraph = active_indexes[request.index_id]

    full_context_list = request.message_history + [request.message]
    full_context: str = "\n".join(full_context_list)

    result = await asyncio.to_thread(get_answer_sync, current_rag, full_context)

    return {"response": result}