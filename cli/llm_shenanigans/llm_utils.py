import os

from dotenv import load_dotenv
from google import genai


def make_llm_query(query: str) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environemnt variable noto set")

    client = genai.Client(api_key=api_key)

    model = "gemma-3-27b-it"

    response = client.models.generate_content(model=model, contents=query)
    if response.text is None:
        raise ValueError("Something wrong happened with the query enhancement")
    if response.usage_metadata is None:
        raise ValueError("Something wrong happened with the query enhancement")
    print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    return response.text
