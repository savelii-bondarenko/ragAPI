from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    index_id: str = Field(..., description="ID vectorstore")
    message: str = Field(..., description="User message")
    message_history: List[str] = Field(default=[], description="Chat history")

class QueryResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    index_id: str