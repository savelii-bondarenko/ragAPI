from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
import logging

from src.core.config import settings

logger = logging.getLogger(__name__)

client = QdrantClient(path=settings.QDRANT_PATH)

def create_vectorDB(documents: list[Document], embedder) -> QdrantVectorStore:
    try:
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embedder,
            url=settings.QDRANT_PATH,
            collection_name=settings.QDRANT_COLLECTION,
            client=client
        )
        logger.info("Successfully created/updated Qdrant vectorstore")
        return vector_store
    except Exception as e:
        logger.critical(f"Failed to create vectorDB: {e}")
        raise e