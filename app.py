from typing import Any

import gradio as gr

from backend.translation import translate, is_dzongkha, DZONGKHA_CODE, ENGLISH_CODE
from backend.llm import query_ollama


def _history_messages(history: list[Any]) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append((role, content))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_message, assistant_message = entry
            if isinstance(user_message, str):
                messages.append(("user", user_message))
            if isinstance(assistant_message, str):
                messages.append(("assistant", assistant_message))
    return messages[-4:]


def process_chat(user_input: str, history: list[Any]) -> str:
    if not user_input.strip():
        return "Please enter a message."

    # 1. Translate Dzongkha questions to English for the reasoning model.
    if is_dzongkha(user_input):
        english_input = translate(user_input, src_lang=DZONGKHA_CODE, tgt_lang=ENGLISH_CODE)
    else:
        english_input = user_input

    english_history = []
    for role, message in _history_messages(history):
        english_message = (
            translate(message, src_lang=DZONGKHA_CODE, tgt_lang=ENGLISH_CODE)
            if is_dzongkha(message)
            else message
        )
        english_history.append({"role": role, "content": english_message})

    # 2. Query local LLM (qwen2.5:3b)
    english_response = query_ollama(english_input, history=english_history)

    # 3. Always reply in Dzongkha, regardless of the input language.
    return translate(english_response, src_lang=ENGLISH_CODE, tgt_lang=DZONGKHA_CODE)


# Pass CSS and a matching palette to the Gradio interface.
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="amber",
        neutral_hue="slate",
    ),
    css="assets/custom.css",
) as demo:
    gr.ChatInterface(
        fn=process_chat,
        title="རྫོང་ཁའི་ ལན་འདེབས་འཕྲུལ་ཆས། (Dzongkha AI Chatbot)",
        description="Ask in Dzongkha or English and get a Dzongkha reply. Runs locally on CPU.",
        examples=[
            ["འབྲུག་གི་རྒྱལ་ས་གར་ཡོདཔ།"],
            ["What is the capital of Bhutan?"],
            ["Tell me about Bhutanese culture."]
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)