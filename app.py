import os
import threading

import torch
from flask import Flask, jsonify, render_template, request
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_GEMMA_MODEL = os.getenv("GEMMA_BASE_MODEL", "google/gemma-2b")
MEANING_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "models", "meaning_adapter")
SENTIMENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "sentiment")

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive",
    2: "No Impact",
    3: "Mixed",
}

app = Flask(__name__)

meaning_model = None
meaning_tokenizer = None
sentiment_model = None
sentiment_tokenizer = None
model_lock = threading.Lock()


def load_sentiment_model():
    """Load the fine-tuned 4-class DistilBERT model."""
    global sentiment_model, sentiment_tokenizer

    print("Loading sentiment model...")
    # The training notebook did not include the tokenizer in the supplied ZIP.
    # It was initialized from this original SST-2 tokenizer, so we reuse it.
    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_DIR,
        num_labels=4,
    )
    sentiment_model.to(device)
    sentiment_model.eval()
    print(f"Sentiment model loaded on {device}.")


def load_meaning_model():
    """Load Gemma-2B plus the supplied 4-projection LoRA adapter."""
    global meaning_model, meaning_tokenizer

    print("Loading poem-meaning model...")
    meaning_tokenizer = AutoTokenizer.from_pretrained(MEANING_ADAPTER_DIR,use_fast=False)

    if meaning_tokenizer.pad_token is None:
        meaning_tokenizer.pad_token = meaning_tokenizer.eos_token

    hf_token = os.getenv("HF_TOKEN") or None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401

            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_GEMMA_MODEL,
                token=hf_token,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("Gemma base model loaded in 4-bit mode.")
        except Exception as exc:
            print(
                "4-bit loading was unavailable; falling back to GPU "
                f"loading. Reason: {exc}"
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_GEMMA_MODEL,
                token=hf_token,
                torch_dtype=torch.float16,
            ).to(device)
    else:
        print(
            "CUDA was not detected. Gemma-2B will be loaded in CPU "
            "full precision. This can require substantial RAM and may be slow."
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_GEMMA_MODEL,
            token=hf_token,
            torch_dtype=torch.float32,
        ).to(device)

    meaning_model = PeftModel.from_pretrained(
        base_model,
        MEANING_ADAPTER_DIR,
    )
    meaning_model.eval()

    print("Poem-meaning model loaded.")


def initialize_models():
    load_sentiment_model()
    load_meaning_model()


def predict_sentiment(poem: str):
    device = next(sentiment_model.parameters()).device

    inputs = sentiment_tokenizer(
        poem,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = sentiment_model(**inputs).logits
        label_id = int(torch.argmax(logits, dim=-1).item())

    return SENTIMENT_LABELS[label_id], label_id


def explain_poem(poem: str):
    prompt = f"""You are an expert at explaining the meaning of the poem. Give an explanation of the poem in
simple words, explaining the main idea of what the author is trying to say.
Generate only the explanation part. Do not repeat yourself by writing the prompt given by user again.
Do not write anything other than the explanation.

Poem:
{poem}

Explanation:"""

    # For quantized models, inputs should be sent to the model's input device.
    try:
        input_device = meaning_model.device
    except Exception:
        input_device = next(meaning_model.parameters()).device

    inputs = meaning_tokenizer(prompt, return_tensors="pt").to(input_device)

    with torch.inference_mode():
        outputs = meaning_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=meaning_tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens, so the UI does not show the prompt.
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    explanation = meaning_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return explanation


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    poem = (data.get("poem") or "").strip()

    if not poem:
        return jsonify({"error": "Please enter a poem."}), 400

    try:
        # The models are kept in memory. A lock prevents simultaneous GPU
        # generation/classification requests from competing for resources.
        with model_lock:
            sentiment, sentiment_id = predict_sentiment(poem)
            explanation = explain_poem(poem)

        return jsonify(
            {
                "explanation": explanation,
                "sentiment": sentiment,
                "sentiment_id": sentiment_id,
            }
        )
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(
            {
                "error": (
                    "Prediction failed. Check the terminal for the full "
                    f"error. Details: {exc}"
                )
            }
        ), 500


if __name__ == "__main__":
    initialize_models()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
