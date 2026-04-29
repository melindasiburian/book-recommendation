import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")
if not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
    raise RuntimeError(
        "OPENROUTER_API_KEY does not look like a valid OpenRouter key. "
        "Expected it to start with 'sk-or-v1-'."
    )

print("Using OPENROUTER_API_KEY from environment.")
print("Key prefix:", OPENROUTER_API_KEY[:12] + "...")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-27b-it:free"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}
MAX_RETRIES = 3


def post_chat(messages: list[dict], reasoning_enabled: bool = True) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.post(
            url=API_URL,
            headers=HEADERS,
            data=json.dumps(
                {
                    "model": MODEL,
                    "messages": messages,
                    "reasoning": {"enabled": reasoning_enabled},
                }
            ),
            timeout=60,
        )

        if response.ok:
            data = response.json()
            if "choices" not in data:
                raise KeyError(f"Expected 'choices' in response, got: {data}")
            return data["choices"][0]["message"]

        print("Status:", response.status_code)
        print("Body:", response.text)
        try:
            error_data = response.json()
        except ValueError:
            error_data = {}

        error_message = error_data.get("error", {}).get("message", "")
        if "No endpoints available matching your guardrail restrictions and data policy" in error_message:
            raise RuntimeError(
                "OpenRouter rejected the request because your privacy/guardrail settings "
                f"do not allow any provider for model '{MODEL}'. Update settings at "
                "https://openrouter.ai/settings/privacy or choose a different model."
            )

        if response.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else 2 ** attempt
            print(f"Rate limited. Retrying in {wait_seconds:.0f} seconds...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

    raise RuntimeError("Request failed after retries")


first_message = post_chat(
    [{"role": "user", "content": "How many r's are in the word 'strawberry'?"}]
)

messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": first_message.get("content"),
        "reasoning_details": first_message.get("reasoning_details"),
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

second_message = post_chat(messages)
print(second_message.get("content"))
