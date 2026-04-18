from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models  # Добавили импорт моделей Qdrant
import logging

from src.core.config import settings

logger = logging.getLogger(__name__)

# 1. ЕДИНСТВЕННЫЙ ГЛОБАЛЬНЫЙ КЛИЕНТ
qdrant_client = QdrantClient(path=settings.QDRANT_PATH)


def create_vectorDB(documents: list[Document], embedder) -> QdrantVectorStore:
    try:
        # 2. ПРОВЕРЯЕМ СУЩЕСТВОВАНИЕ КОЛЛЕКЦИИ
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            logger.info(f"Создаем новую коллекцию: {settings.QDRANT_COLLECTION}")

            # Динамически узнаем размер вектора, который выдает твоя модель
            test_vector = embedder.embed_query("test")
            vector_size = len(test_vector)

            # Создаем пустую коллекцию с нужными параметрами
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

        # 3. ПОДКЛЮЧАЕМСЯ К СУЩЕСТВУЮЩЕЙ БАЗЕ
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=embedder,
        )

        # 4. ДОБАВЛЯЕМ ДОКУМЕНТЫ
        vector_store.add_documents(documents)

        logger.info("Successfully added documents to Qdrant vectorstore")
        return vector_store

    except Exception as e:
        logger.critical(f"Failed to create vectorDB: {e}")
        raise e