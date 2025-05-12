from __future__ import annotations
import argparse, json, os, re, sys, hashlib
from typing import List, Dict, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  MODEL & DEVICE
MODEL_NAME: str = "ElKulako/cryptobert"
MAX_LEN: int   = 256                  # up from 128 to reduce truncation artefacts
DEVICE: str    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧  Loading {MODEL_NAME} on {DEVICE} …")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_model.to(DEVICE).eval()

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ---------------------------------------------------------------------------
# 1.  SNIPPET CLEANER --------------------------------------------------------
_BANNED_STARTERS = (
    "Meanwhile", "However", "On the other hand", "In other news",
    "At the same time", "Elsewhere"
)
ELLIPSIS_RE   = re.compile(r"\.{3}|…")
TIMECODE_RE   = re.compile(r"\b\d{1,2}:\d{2}\b")
VIEW_LIKE_RE  = re.compile(r"\b[\d,.]+[+]?\s+(views?|likes?)\b", re.I)
TRAILER_RE    = re.compile(r"\s*[\-|–|]\s*[^-–|]+$")  # “ – Bloomberg”, etc.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_snippet(snippet: str, aggressive: bool = False) -> str:
    """Strip common SERP artefacts & low‑signal lead‑ins."""
    if not snippet:
        return ""

    snippet = ELLIPSIS_RE.split(snippet)[0]
    snippet = TIMECODE_RE.sub("", snippet)
    snippet = VIEW_LIKE_RE.sub("", snippet)
    snippet = TRAILER_RE.sub("", snippet)

    sents: List[str] = SENT_SPLIT_RE.split(snippet)
    keep: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if aggressive and any(s.startswith(bad) for bad in _BANNED_STARTERS):
            continue
        keep.append(s)

    return re.sub(r"\s+", " ", " ".join(keep)).strip()

# ---------------------------------------------------------------------------
# 2.  INFERENCE HELPERS ------------------------------------------------------
@torch.inference_mode()
def _infer_batch(texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
    """Return [p_neg, p_neu, p_pos] for every text in *texts*."""
    probs_all: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        enc = _tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        logits = _model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().tolist()
        probs_all.extend(probs)
    return probs_all


def _infer_title_snippet(title: str, snippet: str, title_weight: float = 0.5) -> List[float]:
    pt = _infer_batch([title], batch_size=1)[0]
    ps = _infer_batch([snippet or title], batch_size=1)[0]  # fall back to title
    return [title_weight * t + (1 - title_weight) * s for t, s in zip(pt, ps)]

# ---------------------------------------------------------------------------
# 3.  JSON UTILITIES ---------------------------------------------------------

def _hash_row(r: Dict[str, str]) -> str:
    return hashlib.sha1(r["title"].encode("utf-8")).hexdigest()


def process_file(
    in_path: str,
    out_path: str,
    use_separate: bool = True,
    title_weight: float = 0.5,
    conf_thresh: float | None = 0.2,
) -> None:
    rows = json.load(open(in_path, "r", encoding="utf-8"))

    seen = set()  # title‑dedupe
    cleaned_rows = []
    for r in rows:
        h = _hash_row(r)
        if h in seen:
            continue
        seen.add(h)
        cleaned_rows.append(r)
    rows = cleaned_rows

    probs_list: List[List[float]] = []

    if use_separate:
        for r in rows:
            title   = r["title"].strip()
            snippet = clean_snippet(r.get("snippet", ""))
            if len((title + " " + snippet).split()) < 3:
                # Skip very short rows
                probs_list.append([0.33, 0.34, 0.33])
                continue
            probs = _infer_title_snippet(title, snippet, title_weight)
            probs_list.append(probs)
    else:
        texts = [
            f"{r['title']} {clean_snippet(r.get('snippet', ''))}".strip()
            for r in rows
        ]
        probs_list = _infer_batch(texts)

    for r, p in zip(rows, probs_list):
        # Optional confidence thresholding
        label_idx = int(torch.tensor(p).argmax())
        if conf_thresh is not None and abs(p[2] - p[0]) < conf_thresh:
            label_idx = LABEL2ID["neutral"]
        r.update(
            {
                "sentiment": ID2LABEL[label_idx],
                "probs": {"neg": p[0], "neu": p[1], "pos": p[2]},
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), indent=4)


# ---------------------------------------------------------------------------
# 4.  DIRECTORY HELPER -------------------------------------------------------

def batch_dir(
    input_dir: str,
    output_dir: str,
    use_separate: bool,
    title_weight: float,
    conf_thresh: float | None,
):
    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith("_cleaned.json"):
            continue
        in_fp  = os.path.join(input_dir, fn)
        out_fp = os.path.join(output_dir, fn.replace("_cleaned.json", "_scored.json"))
        tqdm.write(f"→ {fn} → {os.path.basename(out_fp)}")
        process_file(
            in_fp,
            out_fp,
            use_separate=use_separate,
            title_weight=title_weight,
            conf_thresh=conf_thresh,
        )


# ---------------------------------------------------------------------------
# 5.  CLI --------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Batch‑score JSON SERP data with cryptobert")
    p.add_argument("--input", required=True, help="Directory containing *_cleaned.json files")
    p.add_argument("--output", required=True, help="Directory to save *_scored.json files")
    p.add_argument("--concat", action="store_true", help="Concatenate title + snippet instead of separate mode")
    p.add_argument("--title_weight", type=float, default=0.5, help="Weight for the title when averaging (0–1)")
    p.add_argument("--conf_thresh", type=float, default=0.2, help="If |p_pos - p_neg| < t, force label neutral; set to -1 to disable")
    args = p.parse_args()

    use_separate = not args.concat
    conf = None if args.conf_thresh is not None and args.conf_thresh < 0 else args.conf_thresh

    batch_dir(
        args.input,
        args.output,
        use_separate=use_separate,
        title_weight=args.title_weight,
        conf_thresh=conf,
    )
    print("🎉  All files scored and saved.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        BTC_INPUT_DIR  = r"Sentiment_Analysis\google scraping\cleaned data\bitcoin_cleaned"
        BTC_OUTPUT_DIR = r"Sentiment_Analysis\BERT runners\sentiment_cryptobert\bitcoin" 
        ETH_INPUT_DIR  = r"Sentiment_Analysis\google scraping\cleaned data\ethereum_cleaned"
        ETH_OUTPUT_DIR = r"Sentiment_Analysis\BERT runners\sentiment_cryptobert\ethereum" 

        batch_dir(
            BTC_INPUT_DIR,
            BTC_OUTPUT_DIR,
            use_separate=True,
            title_weight=0.5,
            conf_thresh=0.2,
        )
        batch_dir(
            ETH_INPUT_DIR,
            ETH_OUTPUT_DIR,
            use_separate=True,
            title_weight=0.5,
            conf_thresh=0.2,
        )
        print("✅  Finished sample run– use --help for CLI mode.")
    else:
        _cli()
