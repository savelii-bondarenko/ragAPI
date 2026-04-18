import os

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore

from typing import TypedDict, Annotated
from langchain_deepseek import ChatDeepSeek

from .utils.tools import TOOLS
from .utils.create_embeddings import shared_embedder

from src.core.config import settings
from src.core.utils.vectorstore import qdrant_client

from dotenv import load_dotenv

load_dotenv()

model = ChatDeepSeek(
    model=settings.LLM_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
).bind_tools(TOOLS)


class State(TypedDict):
    extracted_docs: list[Document]
    messages: Annotated[list[AnyMessage], add_messages]


class RAGGraph:
    """ Initializes the RAGGraph in stateless mode.

    Args:
        index_id (str): Уникальный ID документа (UUID),
                        по которому будет производиться поиск в базе.
    """

    def __init__(self, index_id: str):
        self.model = model
        self.index_id = index_id
        self._tool_node = ToolNode(TOOLS)

        # Используем ЕДИНОГО клиента из vectorstore.py
        self.vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=shared_embedder
        )

    def _retriever_node(self, state: State) -> State:
        """
        Retriever LangGraph node.
        """
        last_message = state["messages"][-1]
        user_query = last_message.content

        search_k = int(os.getenv("VDB_SEARCH_K", 4))

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": settings.VDB_SEARCH_K,
                "filter": qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.document_id",
                            match=qdrant_models.MatchValue(value=self.index_id),
                        )
                    ]
                )
            }
        )

        found_docs = retriever.invoke(user_query)

        return {"extracted_docs": found_docs}

    def _generate_node(self, state: State) -> State:
        """Generation LangGraph node."""
        docs_content = "\n".join(doc.page_content for doc in state["extracted_docs"])
        system_msg = SystemMessage(
            content=f"Context:\n{docs_content}\n\nAnswer the user's question using the context provided."
        )
        messages = [system_msg] + state["messages"]
        response: AIMessage = self.model.invoke(messages)
        return {"messages": [response]}

    def _should_continue_node(self, state: State) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("tools", self._tool_node)

        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "generate")
        graph.add_conditional_edges(
            "generate",
            self._should_continue_node,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph.add_edge("tools", "generate")
        return graph.compile()

    def get_query(self, user_question: str):
        """Send and get text to and from LLM."""
        app = self._build_graph()
        initial_state = {
            "messages": [HumanMessage(content=user_question)],
            "extracted_docs": []
        }
        result = app.invoke(initial_state)
        last_msg = result["messages"][-1]
        return {
            "text": last_msg.content,
        }