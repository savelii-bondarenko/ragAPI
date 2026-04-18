import asyncio
from langchain_core.documents import Document

from .utils.data_reader import read_data
from .utils.split_text import split_text
from .utils.create_embeddings import shared_embedder
from .utils.vectorstore import create_vectorDB

async def prepare_rag_assets(file_path: str, index_id: str):
    extracted_text = await asyncio.to_thread(read_data, file_path)
    splitted_text = await asyncio.to_thread(split_text, extracted_text)

    for doc in splitted_text:
        if not isinstance(doc.metadata, dict):
            doc.metadata = {}
        doc.metadata["document_id"] = index_id

    vector_db = await asyncio.to_thread(create_vectorDB, splitted_text, shared_embedder)

    return True