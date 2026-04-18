from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import logging
import os


logger = logging.getLogger(__name__)

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
client = QdrantClient(path=db_path)

def create_vectorDB(embeddings: np.ndarray) -> faiss.Index:
    """Create or update Qdrant vector store from documents.

    Args:
        documents (list[Document]): splitted documents.
        embedder: object for creating embedding vectors.

    Returns:
        QdrantVectorStore: Vector store object.

    Raises:
        Exception: if something went wrong.
    """
    collection_name = "demo_collection"
    try:
       vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embedder,
            url=db_path,
            collection_name=collection_name,
            client=client
        )

       logger.info("Successfully created/updated Qdrant vectorstore")
       return vector_store
    except Exception as e:
       logger.critical(f"Failed to create vectorDB: {e}")
       raise e

