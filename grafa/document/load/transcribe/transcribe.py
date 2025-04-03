"""Module for transcribing documents."""

from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langfuse.decorators import langfuse_context, observe
from PIL import Image

from ..utils import image_to_base64
from .prompt import OCR_PROMPT
from .utils import transcription_extractor


@observe
async def get_transcription(image: Image.Image, llm: Runnable) -> str:
    """Perform OCR on an image and return the transcription."""
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    t = HumanMessagePromptTemplate.from_template(
        template=[
            {"type": "text", "text": "{image_context}"},
            {
                "type": "image_url",
                "image_url": "{image_path}",
                "detail": "{detail_parameter}",
            },
        ]
    )

    image_base64 = image_to_base64(image)
    chat_message = t.format(
        image_context=OCR_PROMPT,
        image_path=image_base64,
        detail="high",
    )
    return transcription_extractor(
        await llm.ainvoke(
            [chat_message], config={"callbacks": [langfuse_handler]}
        ).content
    )
