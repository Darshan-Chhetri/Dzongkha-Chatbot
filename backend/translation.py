import re
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_NEW_TOKENS = 128
NUM_BEAMS = 2
print("Initializing NLLB-200 translation model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
_tokenizer_lock = threading.Lock()

DZONGKHA_CODE = "dzo_Tibt"
ENGLISH_CODE = "eng_Latn"


def is_dzongkha(text: str) -> bool:
    """Detects Dzongkha/Tibetan script range (U+0F00 to U+0FFF)."""
    return bool(re.search(r'[\u0F00-\u0FFF]', text))

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translates text locally using Meta's NLLB-200 model."""
    if not text.strip():
        return ""

    for language_code in (src_lang, tgt_lang):
        token_id = tokenizer.convert_tokens_to_ids(language_code)
        if tokenizer.convert_ids_to_tokens(token_id) != language_code:
            raise ValueError(f"Unsupported NLLB language code: {language_code}")

    # NLLB tokenizers carry source language as mutable state.
    with _tokenizer_lock:
        tokenizer.src_lang = src_lang
        inputs = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]