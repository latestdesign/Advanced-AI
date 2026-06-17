from PIL import Image

from data.processors import get_image_string

import torch
import logging

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

logger = logging.getLogger(__name__)



def load_model(checkpoint: str) -> None:
    """
    Load trained VisionLanguageModel from checkpoint.
    Idempotent: if already loaded, does nothing.
    """

    global _model, _tokenizer, _image_processor, _device, _cfg

    if _model is not None:
        return

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    logger.info("Loading checkpoint from %s on %s", checkpoint, _device)

    _model = VisionLanguageModel.from_pretrained(checkpoint).to(_device)
    _model.eval()

    _cfg = _model.cfg

    _tokenizer = get_tokenizer(
        _cfg.lm.tokenizer,
        _cfg.image_token
    )

    _image_processor = get_image_processor(
        _cfg.vit.img_size
    )

    logger.info("Model loaded")

# Ces variables sont initialisées dans load_model(checkpoint)
_model = None
_tokenizer = None
_image_processor = None
_device = None
_cfg = None


def generate(
    messages: list[dict],
    image: Image.Image | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate an answer using the loaded VisionLanguageModel.

    Args:
        messages: list of dicts like [{"role": "user", "content": "..."}]
        image: PIL image or None
        max_new_tokens: maximum number of new tokens
        temperature: generation temperature

    Returns:
        str: generated answer
    """

    global _model, _tokenizer, _image_processor, _device, _cfg

    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model(checkpoint) first.")

    if image is None:
        raise ValueError("This VisionLanguageModel needs an image. image cannot be None.")

    # Process image
    image = image.convert("RGB")
    pixel_values = _image_processor(image).unsqueeze(0).to(_device)

    # Image placeholder tokens
    image_string = get_image_string(
        _cfg.projector.image_token_length,
        _cfg.image_token
    )

    # Find the last user message
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"),
        default=-1
    )

    if last_user_idx == -1:
        raise ValueError("messages must contain at least one user message.")

    # Build messages for tokenizer
    formatted_messages = []

    for i, m in enumerate(messages):
        role = m["role"]
        content = m["content"]

        # Put image tokens only before the last user message
        if role == "user" and i == last_user_idx:
            content = image_string + content

        formatted_messages.append({
            "role": role,
            "content": content
        })

    encoded = _tokenizer.apply_chat_template(
        [formatted_messages],
        tokenize=True,
        add_generation_prompt=True
    )

    input_ids = torch.tensor(encoded).to(_device)
    attention_mask = torch.ones_like(input_ids)

    greedy = temperature <= 0.0

    with torch.no_grad():
        gen = _model.generate(
            input_ids,
            pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature if not greedy else 1.0,
        )

    text = _tokenizer.batch_decode(
        gen,
        skip_special_tokens=True
    )[0]

    return text.strip()