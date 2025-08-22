# deepface/tools/ai_image_editor/chatbot.py

"""
Chatbot helper for AI Image Editor.
Uses OpenAI's API (gpt-4o-mini) if OPENAI_API_KEY is set.
"""

import os
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are the assistant for an image-and-face toolbox. "
    "Start by greeting the user and asking what they want to do today. "
    "From their reply, choose the closest match among these features: "
    "Image Processing (Detect Faces, Resize, Rescale, Rotate, Masked Image, Smart Blur, Remove Faces); "
    "Face Verification (Photo Upload compares two high-quality photos with OpenFace model, "
    "Webcam Verification compares a reference photo with a live webcam shot using VGG-Face); "
    "Face Analysis (returns age, gender, emotion, and race). "
    "Confirm your choice in one sentence, briefly explain what it does, "
    "and tell the user which page/button to open next. "
    "If two features could work, name both, give a one-line comparison, and let the user choose. "
    "If the request is unclear, ask a follow-up question instead of guessing. "
    "Keep replies short, friendly, and jargon-free unless the user asks for technical detail."
)

def is_enabled() -> bool:
    """Check if chatbot can run (requires OPENAI_API_KEY)."""
    return bool(os.getenv("OPENAI_API_KEY"))

def get_response(user_input: str, conversation_history=None) -> str:
    """
    Generate a chatbot reply using OpenAI API.
    
    Args:
        user_input: str, the latest user message.
        conversation_history: list of dicts [{'role': 'user'/'assistant', 'content': str}], optional.
    
    Returns:
        str: chatbot reply.
    """
    if not is_enabled():
        return "Chatbot unavailable: please set your OPENAI_API_KEY environment variable."

    if conversation_history is None:
        conversation_history = []

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages.extend(conversation_history)
        messages.append({'role': 'user', 'content': user_input})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def response(user_input: str, conversation_history=None) -> str:
    """
    Alias for get_response to match the working editor.py expectations.
    """
    return get_response(user_input, conversation_history)
