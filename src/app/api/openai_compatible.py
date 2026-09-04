import time
import uuid

from fastapi import APIRouter
from app.schemas.openai_schemas import ChatCompletionRequest
from app.services.langgraph_service import generate_answer

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI Compatible API"]
)

@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "urology-rag-doctor",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "urology-team"
            },
            {
                "id": "urology-rag-patient",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "urology-team"
            }
        ]
    }


@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest):

    user_message = request.messages[-1].content

    if request.model == "medical-rag-doctor":
        user_type = "doctor"

    elif request.model == "medical-rag-patient":
        user_type = "patient"

    else:
        raise ValueError(f"Unknown model: {request.model}")

    print("MODEL:", request.model)
    print("ROLE:", user_type)

    answer = generate_answer(
        question=user_message,
        user_type=user_type
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }