from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import os
import tempfile
import asyncio

from src.api.schemas import QueryRequest, QueryResponse, UploadResponse
from src.core.graph_logic import RAGGraph
from src.core.engine import prepare_rag_assets


router = APIRouter()


@router.post("/index", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    index_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        await prepare_rag_assets(tmp_path, index_id)

        return UploadResponse(index_id=index_id)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    context = "\n".join(request.message_history + [request.message])

    rag_instance = RAGGraph(index_id=request.index_id)

    result = await asyncio.to_thread(rag_instance.get_query, context)
    return QueryResponse(answer=result["text"])