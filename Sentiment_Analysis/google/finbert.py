#!/usr/bin/env python
# score_finbert_serp.py
# ----------------------------------------------------------
# Batch-score Google-SERP news snippets with FinBERT.
# Adds fields:  "sentiment"  and  "probs"  to each row.

import os, re, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# 0.  MODEL & DEVICE
MODEL_NAME = "ProsusAI/finbert"           # change to your fine-tuned checkpoint if needed
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE).eval()

id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ------------------------------------------------------------------
# 1.  SNIPPET SCRUBBER
BANNED_STARTERS = (
    "Meanwhile", "However", "But", "On the other hand",
    "In other news", "At the same time", "Elsewhere"
)

ELLIPSIS_RE   = re.compile(r"\.{3}|…")
TIMECODE_RE   = re.compile(r"\b\d{1,2}:\d{2}\b")
VIEW_LIKE_RE  = re.compile(r"\b[\d,.]+[+]?\s+(views?|likes?)\b", re.I)
TRAILER_RE    = re.compile(r"\s*[\-|–|]\s*[^-–|]+$")   # “ – Bloomberg” etc.

def clean_snippet(snippet: str) -> str:
    """Strip SERP artefacts & trailing bearish clauses."""
    if not snippet:
        return snippet

    snippet = ELLIPSIS_RE.split(snippet)[0]
    snippet = TIMECODE_RE.sub("", snippet)
    snippet = VIEW_LIKE_RE.sub("", snippet)
    snippet = TRAILER_RE.sub("", snippet)

    sents, keep = re.split(r"(?<=[.!?])\s+", snippet), []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if any(s.startswith(bad) for bad in BANNED_STARTERS):
            continue
        keep.append(s)

    return re.sub(r"\s+", " ", " ".join(keep)).strip()

# ------------------------------------------------------------------
# 2.  INFERENCE HELPERS
@torch.inference_mode()
def infer_batch(texts, batch_size=32):
    """Return list of [p_neg, p_neu, p_pos] for each text."""
    probs_all = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i+batch_size],
            padding=True, truncation=True, max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().tolist()
        probs_all.extend(probs)
    return probs_all

def infer_title_snippet(title: str, snippet: str):
    """Score title and snippet separately, then average the probabilities."""
    p_title   = infer_batch([title], batch_size=1)[0]
    p_snippet = infer_batch([snippet], batch_size=1)[0]
    probs = [(t + s) / 2 for t, s in zip(p_title, p_snippet)]
    return probs  # same order: [neg, neu, pos]

# ------------------------------------------------------------------
# 3.  FILE PROCESSING
def process_file(in_path, out_path, use_separate=False):
    with open(in_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    texts, probs_list = [], []

    if use_separate:
        # title + snippet scored independently
        for r in rows:
            title   = r["title"].strip()
            snippet = clean_snippet(r.get("snippet", ""))
            probs   = infer_title_snippet(title, snippet)
            probs_list.append(probs)
    else:
        # single pass on concatenated string
        texts = [
            f"{r['title']} {clean_snippet(r.get('snippet', ''))}".strip()
            for r in rows
        ]
        probs_list = infer_batch(texts, batch_size=32)

    for r, p in zip(rows, probs_list):
        sentiment = id2label[int(torch.tensor(p).argmax())]
        r.update({"sentiment": sentiment,
                  "probs": {"neg": p[0], "neu": p[1], "pos": p[2]}})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)

def batch_dir(input_dir, output_dir, use_separate=False):
    for fn in os.listdir(input_dir):
        if fn.endswith("_cleaned.json"):
            in_fp  = os.path.join(input_dir, fn)
            out_fp = os.path.join(
                output_dir, fn.replace("_cleaned.json", "_scored.json")
            )
            tqdm.write(f"→  {fn}  →  {os.path.basename(out_fp)}")
            process_file(in_fp, out_fp, use_separate=use_separate)

# ------------------------------------------------------------------
# 4.  RUN
if __name__ == "__main__":
    BTC_INPUT_DIR  = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\google_search_data\cleaned\bitcoin_cleaned"
    BTC_OUTPUT_DIR = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\google_search_data\sentiment\bitcoin" 
    ETH_INPUT_DIR  = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\google_search_data\cleaned\ethereum_cleaned"
    ETH_OUTPUT_DIR = r"C:\Users\user\Desktop\Uni_work\year_3\Sem_2\DIA\project code\sentiment_data\google_search_data\sentiment\ethereum" 

    # Set to True to enable the safer title+snippet averaging method
    USE_SEPARATE_MODE = True

    batch_dir(BTC_INPUT_DIR, BTC_OUTPUT_DIR)
    batch_dir(ETH_INPUT_DIR, ETH_OUTPUT_DIR)

    print("🎉  All files scored and saved.")
