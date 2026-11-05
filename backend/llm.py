import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"


def query_ollama(prompt: str, history: list[dict[str, str]] | None = None) -> str:
    """Sends a structured conversation to the local Ollama chat API."""
    system_prompt = (
        "You are a helpful assistant answering questions about Bhutan and general topics. "
        "Answer in simple, clear English so the response can be translated into Dzongkha. "
        "Be accurate, distinguish facts from uncertainty, and do not invent details. "
        "If a question is ambiguous, ask one concise clarifying question."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"Unable to get a response from local Ollama: {error}") from error

    if not isinstance(data, dict):
        raise RuntimeError("Local Ollama returned an invalid response.")
    message = data.get("message")
    result = message.get("content") if isinstance(message, dict) else None
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError("Local Ollama returned an empty or invalid response.")
    return result.strip()