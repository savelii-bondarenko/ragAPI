from langchain_core.documents import Document
from faiss import Index

import asyncio

from .utils.data_reader import read_data
from .utils.split_text import split_text
from .utils.create_embeddings import shared_embedder
from .utils.vectorstore import create_vectorDB


async def prepare_rag_assets(file_path: str):
    """Prepare data for RAG

        Args:
            file_path (str): Path to the file

        Returns:
            splitted_text (str): splitted text for chunks.

            embedder (Embedder): embedder class for create embeddings.

            vector_db (faiss.Index): vector database.
    """
    extracted_text = await asyncio.to_thread(read_data, file_path)
    splitted_text = await asyncio.to_thread(split_text, extracted_text)
    embeddings = await asyncio.to_thread(shared_embedder.make_embeddings, splitted_text)
    vector_db = await asyncio.to_thread(create_vectorDB, embeddings)

    return splitted_text, shared_embedder, vector_db