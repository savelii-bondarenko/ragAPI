from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import os
import tempfile
import asyncio

from src.api.schemas import QueryRequest, QueryResponse, UploadResponse
from src.core.engine import prepare_rag_assets
from src.core.graph import RAGGraph
from src.core import shared_embedder

router = APIRouter()
active_indexes = {}


@router.post("/index", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    index_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        splitted_text, _, vector_db = await asyncio.to_thread(prepare_rag_assets, tmp_path)

        rag_instance = RAGGraph(splitted_text, shared_embedder, vector_db)
        active_indexes[index_id] = rag_instance
        return UploadResponse(index_id=index_id)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if request.index_id not in active_indexes:
        raise HTTPException(status_code=404, detail="Index not found")

    rag = active_indexes[request.index_id]
    context = "\n".join(request.message_history + [request.message])

    result = await asyncio.to_thread(rag.get_query, context)
    return QueryResponse(answer=result["text"])