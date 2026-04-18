from FlagEmbedding import FlagModel
from langchain_core.embeddings import Embeddings
import logging

logger = logging.getLogger(__name__)


class Embedder(Embeddings):
    """Embedder class compatible with LangChain."""

    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs):
        """Initializes the Embedder.

        Args:
            model_name (str): The name of the model to use.
            **kwargs: Additional parameters for the FlagModel.
        """
        logger.info(f"Initializing Embedder with model: {model_name}")
        self.model = FlagModel(model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts (list[str]): List of texts to embed.

        Returns:
            list[list[float]]: List of embeddings.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text (str): Query text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        embedding = self.model.encode([text])
        return embedding[0].tolist()


shared_embedder = Embedder(
    query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query",
    use_fp16=True
)