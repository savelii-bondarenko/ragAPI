# AI RAG Microservice (ragAPI)

A high-performance, stateless **Retrieval-Augmented Generation (RAG)** microservice. This service is designed to act as an AI worker for external backends (such as C# ASP.NET Core), providing document indexing, semantic search, and intelligent question-answering using AI agents.

## 🚀 Key Features

* **Multi-Document Indexing:** Upload multiple files (PDF, TXT, etc.) simultaneously and group them under a single `index_id`.
* **Stateless Architecture:** No session state is stored in memory, making the service easy to scale and resilient to restarts.
* **Vector Search:** Powered by **Qdrant** in local storage mode for fast and reliable document retrieval.
* **AI Agents with Tools:** Built using **LangGraph**, allowing the LLM to use external tools (like calculators) to provide accurate answers.
* **Advanced Embeddings:** Uses the `BAAI/bge-m3` model for high-quality multilingual vectorization.
* **Secure API:** Protected via `X-API-Key` header authentication.

## 🛠 Tech Stack

* **Language:** Python 3.12
* **Web Framework:** FastAPI
* **Orchestration:** LangGraph / LangChain
* **Vector Database:** Qdrant
* **LLM:** DeepSeek (via LangChain-OpenAI compatible interface)
* **Embeddings:** FlagEmbedding (BGE-M3)
* **Environment Management:** `uv` (recommended) or `pip`

## 📋 Prerequisites

* Python 3.12.
* A valid **DeepSeek API Key**.
* [uv](https://github.com/astral-sh/uv) installed (for faster dependency management).

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ragapi.git
    cd ragapi
    ```

2.  **Create a virtual environment and install dependencies:**
    Using `uv`:
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install -r pyproject.toml
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    # --- LLM Settings ---
    DEEPSEEK_API_KEY=your_deepseek_api_key_here
    LLM_MODEL=deepseek-chat

    # --- Vector DB Settings ---
    QDRANT_PATH=./qdrant_data
    QDRANT_COLLECTION=all_documents
    VDB_SEARCH_K=4

    # --- RAG Settings ---
    CHUNK_SIZE=512
    CHUNK_OVERLAP=100
    ```

4.  **Run the server:**
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

## 📡 API Endpoints

### 1. Index Documents
`POST /index`
Uploads multiple files, processes them, and stores them in the vector database under a unique session ID.

* **Headers:** `X-API-Key: <your_secret_key>`
* **Payload:** `multipart/form-data` (Field name: `files`)
* **Response:**
    ```json
    {
      "index_id": "064a5207-8705-4b00-ba54-a0b03ac1a1ff"
    }
    ```

### 2. Query AI
`POST /query`
Performs semantic search across the documents linked to the `index_id` and generates a response.

* **Headers:** `X-API-Key: <your_secret_key>`
* **Payload:**
    ```json
    {
      "index_id": "064a5207-8705-4b00-ba54-a0b03ac1a1ff",
      "message": "What is the secret code mentioned in the documents?",
      "message_history": []
    }
    ```
* **Response:**
    ```json
    {
      "answer": "The secret code is 1234, as found in the document..."
    }
    ```

## 🧪 Testing

You can use the provided `test_client.py` to verify the installation. It automatically creates test files, uploads them as a batch, and queries the AI.

```bash
python test_client.py
```

## 📂 Project Structure

* `app.py` - Application entry point and FastAPI initialization.
* `src/api/` - API routes and Pydantic request/response schemas.
* `src/core/` - Core logic, including LangGraph orchestration (`graph_logic.py`) and RAG engine.
* `src/core/utils/` - Utility modules for document reading, text splitting, and vector storage.
* `qdrant_data/` - (Auto-generated) Local storage for the Qdrant vector database.

***
