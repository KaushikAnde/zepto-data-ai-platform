
import os
import json
from pathlib import Path
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from prompt import build_prompt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

# Default graded mode = MOCK
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# --------------------------------------------------
# Pydantic output schema
# --------------------------------------------------

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float
    retrieved_context: str


# --------------------------------------------------
# Load embedding model + ChromaDB
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="zepto_policies"
)


# --------------------------------------------------
# Optional real LLM helper
# Used only when MOCK_LLM=0
# --------------------------------------------------

def call_real_llm(prompt: str) -> str:
    """
    Optional real-LLM extension.

    To use:
    1. Set MOCK_LLM=0
    2. Install groq
    3. Set GROQ_API_KEY
    """

    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "Groq is not installed. Run: pip install groq"
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0"
        )

    groq_client = Groq(api_key=api_key)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def generate_validated_real_answer(prompt, sources):
    """
    Retry up to 2 additional times if the LLM output
    does not match the required JSON schema.
    """

    current_prompt = prompt

    for attempt in range(3):

        raw_output = call_real_llm(current_prompt)

        try:
            result = AnswerResponse.model_validate_json(raw_output)
            return result

        except Exception:

            current_prompt += """

IMPORTANT:
Your previous response did not match the required JSON schema.

Return ONLY valid JSON in exactly this structure:

{
  "answer": "your answer",
  "sources": ["source_id"],
  "confidence": 0.9
}

Do not include Markdown or any text outside the JSON.
"""

    return AnswerResponse(
        answer="ERROR: Real LLM failed to produce valid structured output.",
        sources=sources,
        confidence=0.0
    )


# --------------------------------------------------
# Node 1: classify_intent
# --------------------------------------------------

def classify_intent(state: AssistantState):

    query = state["query"]

    if MOCK_LLM:

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:

        classification_prompt = f"""
Classify the following query into exactly one category:

policy_question
general_question

A policy question is about Zepto delivery, returns, refunds,
membership, tracking, cancellation, gift cards or support hours.

Query:
{query}

Return only the category name.
"""

        result = call_real_llm(classification_prompt).lower()

        if "policy_question" in result:
            intent = "policy_question"
        else:
            intent = "general_question"

    return {
        "intent": intent
    }


# --------------------------------------------------
# Node 2: retrieve_and_answer
# --------------------------------------------------

def retrieve_and_answer(state: AssistantState):

    query = state["query"]

    # Embed user query
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    # Retrieve top 3 chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    context = "\n\n".join(documents)

    if MOCK_LLM:

        # Required deterministic mock output
        top_chunk = documents[0]

        top_chunk_snippet = top_chunk[:200].replace(
            "\n", " "
        )

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=ids,
            confidence=1.0
        )

    else:

        prompt = build_prompt(
            question=query,
            context=context
        )

        prompt += f"""

Return the final response as valid JSON with this schema:

{{
    "answer": "string",
    "sources": {json.dumps(ids)},
    "confidence": 0.0
}}
"""

        response = generate_validated_real_answer(
            prompt,
            ids
        )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "retrieved_context": context
    }


# --------------------------------------------------
# Node 3: direct_answer
# --------------------------------------------------

def direct_answer(state: AssistantState):

    query = state["query"]

    if MOCK_LLM:

        response = AnswerResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0
        )

    else:

        prompt = f"""
You are a Zepto customer support assistant.

The following question does not require policy retrieval.

Question:
{query}

Return ONLY JSON:

{{
    "answer": "short answer",
    "sources": [],
    "confidence": 0.9
}}
"""

        response = generate_validated_real_answer(
            prompt,
            []
        )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


# --------------------------------------------------
# Conditional routing
# --------------------------------------------------

def route_intent(state: AssistantState):

    return state["intent"]


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(AssistantState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.add_edge(
    START,
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()


# --------------------------------------------------
# Helper used later by FastAPI
# --------------------------------------------------

def ask_question(query: str) -> AnswerResponse:

    result = graph.invoke({
        "query": query
    })

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
