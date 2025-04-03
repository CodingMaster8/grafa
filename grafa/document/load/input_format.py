"""Module for loading documents into the knowledge graph."""
import io
import os
import tempfile
from functools import wraps
from typing import Callable, Dict

import fsspec
import markdown
from pdf2image import convert_from_path
from PIL import Image
from striprtf.striprtf import rtf_to_text

from .transcribe import get_transcription

# Dictionary to store file format mappings
FILE_PROCESSORS: Dict[str, Callable] = {}


def file_processor(*extensions: str):
    """Register file processing functions.

    Parameters
    ----------
    *extensions : str
        File extensions that this processor can handle

    Returns
    -------
    Callable
        The decorated function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Register the function for each extension
        for ext in extensions:
            FILE_PROCESSORS[ext.lower()] = func

        return wrapper

    return decorator


@file_processor("txt")
async def process_txt(file_path: str) -> str:
    """Process a text file."""
    with fsspec.open(file_path) as f:
        text = f.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
    return text


@file_processor("md", "markdown")
async def process_markdown(file_path: str) -> str:
    """Process a markdown file."""
    with fsspec.open(file_path) as f:
        md_content = f.read()
        if isinstance(md_content, bytes):
            md_content = md_content.decode("utf-8")
    return markdown.markdown(md_content)


@file_processor("rtf")
async def process_rtf(file_path: str) -> str:
    """Process an RTF file."""
    with fsspec.open(file_path) as f:
        rtf_content = f.read()
        if isinstance(rtf_content, bytes):
            rtf_content = rtf_content.decode("utf-8")
    return rtf_to_text(rtf_content)


# pptx and docx should be transformed into pdf first using libreoffice or something similar


@file_processor("pdf")
async def process_pdf(file_path: str) -> str:
    """Process a PDF file."""
    with fsspec.open(file_path, "rb") as f:
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_file.write(await f.read())
        temp_file.close()
        images = convert_from_path(temp_file.name, dpi=200, fmt="jpeg")
        os.unlink(temp_file.name)
    page_texts = []
    for i, image in enumerate(images):
        text = await get_transcription(image)
        page_texts.append(f"<page>{text}</page>")
    return "\n".join(page_texts)


@file_processor("png", "jpg", "jpeg", "svg", "bmp", "tiff", "gif")
async def process_image_file(file_path: str) -> str:
    """Process an image file."""
    with fsspec.open(file_path, "rb") as f:
        image = Image.open(io.BytesIO(await f.read()))
    return await get_transcription(image)
