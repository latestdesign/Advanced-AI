import base64, io, logging, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from PIL import Image

import model
from schemas import ChatMessage, ChatRequest, ChatResponse

logging.basicConfig(level=logging.INFO)

# Checkpoint for model loading
CHECKPOINT = "best_step60"
# `lifespan` is FastAPI's startup/shutdown hook. Anything before `yield` runs
# ONCE, before the server starts accepting requests; anything after `yield`
# runs on shutdown. We load the model here so the first incoming request does
# NOT pay the multi-second model-loading tax.
@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model(CHECKPOINT)
    yield


app = FastAPI(title="VLM Chatbot API", lifespan=lifespan)
@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "VLM Chatbot API is running",
        "documentation": "/docs",
        "health": "/health",
    }

# A GET endpoint — no body, just returns a JSON object.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def truncate_history(messages: list[ChatMessage], max_turns: int) -> list[ChatMessage]:
    """Keep any leading system message + the last 2*max_turns user/assistant entries."""
    system_prefix = [m for m in messages[:1] if m.role == "system"]
    rest = messages[len(system_prefix):]
    return system_prefix + rest[-(2 * max_turns):]


# A POST endpoint — FastAPI parses the JSON body into a ChatRequest for us.
# If the body is invalid (wrong types, out-of-range temperature, etc.), FastAPI
# returns a 422 response automatically and our function is never called.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        # Validation that Pydantic can't express: a non-empty list. We raise
        # an HTTPException, which FastAPI turns into a proper HTTP error response.
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    # Decode the optional image from base64 → bytes → PIL.
    image: Image.Image | None = None
    if req.image_base64:
        try:
            raw = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {e}") from e

    # Trim, unpack Pydantic to plain dicts, generate, measure how long it took.
    trimmed = truncate_history(req.messages, req.max_turns)
    messages_for_model = [{"role": m.role, "content": m.content} for m in trimmed]

    start = time.perf_counter()
    reply = model.generate(
        messages=messages_for_model,
        image=image,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return ChatResponse(reply=reply, generation_time_ms=elapsed_ms)