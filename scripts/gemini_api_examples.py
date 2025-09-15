"""
Checkout this documents for installation: https://ai.google.dev/gemini-api/docs/quickstart?lang=python'
Refer link https://ai.google.dev/gemini-api/docs/api-key to get your Gemini API Key
"""

import asyncio
import ast
from google import genai
from google.genai import types
import PIL.Image

# Setup Gemini client
GEMINI_API_KEY = "ADD_YOUR_TOKEN_HERE"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
CLIENT = genai.Client(api_key=GEMINI_API_KEY)


def check_smoking(image_url: str = None):
    """Check if a person is smoking or not.
    Args:
        image_url (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert image_url is not None, "URL is required"
    image = PIL.Image.open(image_url)
    response = CLIENT.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=["Is this person smoking? Confidence score?", image])
    print(response.text)
    return response.text


def check_using_mobile_phone(image_url: str = None):
    """Check if a person is using a mobile phone or not.
    Args:
        image_url (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert image_url is not None, "URL is required"
    image = PIL.Image.open(image_url)
    response = CLIENT.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=["Is this person using a mobile phone? Confidence score?", image])
    print(response.text)
    return response.text
    

if __name__ == "__main__":
    check_smoking(image_url="test.jpg")
    # check_using_mobile_phone(image_url="test.jpg")
    pass