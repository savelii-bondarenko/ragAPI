from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
import uuid
import os
import tempfile
import asyncio

from src.api.schemas import QueryRequest, QueryResponse, UploadResponse
from src.core.graph_logic import RAGGraph
from src.core.engine import prepare_rag_assets
from src.core.config import settings

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Access denied")
    return api_key


@router.post("/index", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
           await prepare_rag_assets(tmp_path, session_id)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return UploadResponse(index_id=session_id)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    context = "\n".join(request.message_history + [request.message])

    rag_instance = RAGGraph(index_id=request.index_id)

    result = await asyncio.to_thread(rag_instance.get_query, context)
    return QueryResponse(answer=result["text"])