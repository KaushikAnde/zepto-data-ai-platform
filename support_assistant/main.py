
from fastapi import FastAPI
from pydantic import BaseModel

from graph import ask_question, AnswerResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant API is running"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    return ask_question(request.query)
