import os
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import anthropic


PROMPTS_PATH = "data/prompts.parquet"
QUESTIONS_PATH = "data/questions.parquet"
OUT_CSV = "results_sonnet45_batch.csv"

PROMPT_TEXT_COL = "prompt"
IS_JB_COL = "jailbreak"
PLATFORM_COL = "platform"
SOURCE_COL = "source"

QUESTION_TEXT_COL = "prompt"
CATEGORY_COL = "category"

TARGET_CATEGORIES = [
    "Illegal Activity",
    "Hate Speech",
    "Malware",
    "Physical Harm",
    "Financial Advice",
    "Fraud",
]
QUESTIONS_PER_CATEGORY = 10
JAILBREAK_SAMPLE_SIZE = 50
SEED = 42

MODEL_NAME = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 400

CHUNK_SIZE = 200
POLL_SECONDS = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_input_text(jb: Optional[str], q: str) -> str:
    q = q.strip()
    if jb is None:
        return q
    return f"{jb.strip()}\n\nUser question:\n{q}"


def chunked(lst: List[dict], n: int) -> Iterator[List[dict]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY environment variable.")
    return anthropic.Anthropic(api_key=api_key)


def sample_questions(questions_df: pd.DataFrame) -> pd.DataFrame:
    q_all = questions_df[questions_df[CATEGORY_COL].isin(TARGET_CATEGORIES)].copy()
    q_all = q_all.drop_duplicates(subset=[CATEGORY_COL, QUESTION_TEXT_COL]).reset_index(drop=True)

    parts = []
    for cat in TARGET_CATEGORIES:
        qcat = q_all[q_all[CATEGORY_COL] == cat]
        if len(qcat) < QUESTIONS_PER_CATEGORY:
            raise ValueError(f"Not enough questions in {cat}: have {len(qcat)} need {QUESTIONS_PER_CATEGORY}")
        parts.append(qcat.sample(n=QUESTIONS_PER_CATEGORY, random_state=SEED))

    qf = pd.concat(parts).reset_index(drop=True)
    qf["question_key"] = (
        qf[CATEGORY_COL].astype(str) + "||" + qf[QUESTION_TEXT_COL].astype(str)
    ).apply(stable_hash_id)
    return qf


def sample_jailbreaks(prompts_df: pd.DataFrame) -> pd.DataFrame:
    jb_df = prompts_df[(prompts_df[IS_JB_COL] == True) & (prompts_df[PROMPT_TEXT_COL].notna())].copy()
    jb_df["jailbreak_key"] = jb_df[PROMPT_TEXT_COL].astype(str).apply(stable_hash_id)

    n = min(JAILBREAK_SAMPLE_SIZE, len(jb_df))
    return jb_df.sample(n=n, random_state=SEED).reset_index(drop=True)


def build_requests(prompts_df: pd.DataFrame, questions_df: pd.DataFrame) -> Tuple[List[dict], Dict[str, dict]]:
    qf = sample_questions(questions_df)
    jb_sample = sample_jailbreaks(prompts_df)

    requests: List[dict] = []
    meta: Dict[str, dict] = {}

    for _, qrow in qf.iterrows():
        cat = str(qrow[CATEGORY_COL])
        qtext = str(qrow[QUESTION_TEXT_COL])
        qkey = str(qrow["question_key"])

        custom_id = f"baseline_{qkey}"
        user_text = build_input_text(None, qtext)

        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": MODEL_NAME,
                    "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": user_text}],
                },
            }
        )

        meta[custom_id] = {
            "timestamp_utc": utc_now_iso(),
            "condition": "baseline",
            "category": cat,
            "question_key": qkey,
            "question_prompt": qtext,
            "jailbreak_key": None,
            "jailbreak_prompt": None,
            "platform": None,
            "source": None,
            "model": MODEL_NAME,
        }

    for _, qrow in qf.iterrows():
        cat = str(qrow[CATEGORY_COL])
        qtext = str(qrow[QUESTION_TEXT_COL])
        qkey = str(qrow["question_key"])

        for _, prow in jb_sample.iterrows():
            jbtext = str(prow[PROMPT_TEXT_COL])
            jbkey = str(prow["jailbreak_key"])
            platform = str(prow[PLATFORM_COL]) if PLATFORM_COL in prow and pd.notna(prow[PLATFORM_COL]) else None
            source = str(prow[SOURCE_COL]) if SOURCE_COL in prow and pd.notna(prow[SOURCE_COL]) else None

            custom_id = f"jb_{qkey}_{jbkey}"
            user_text = build_input_text(jbtext, qtext)

            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL_NAME,
                        "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": user_text}],
                    },
                }
            )

            meta[custom_id] = {
                "timestamp_utc": utc_now_iso(),
                "condition": "jailbreak",
                "category": cat,
                "question_key": qkey,
                "question_prompt": qtext,
                "jailbreak_key": jbkey,
                "jailbreak_prompt": jbtext,
                "platform": platform,
                "source": source,
                "model": MODEL_NAME,
            }

    print(f"[INFO] Built {len(requests)} requests total.")
    return requests, meta


def ensure_csv(path: str) -> None:
    if os.path.exists(path):
        return

    cols = [
        "run_id",
        "timestamp_utc",
        "condition",
        "category",
        "question_key",
        "question_prompt",
        "jailbreak_key",
        "jailbreak_prompt",
        "platform",
        "source",
        "model",
        "response_text",
        "input_tokens",
        "output_tokens",
        "error",
        "judge_label",
        "custom_id",
        "batch_id",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8")


def extract_text_from_message(msg: Any) -> str:
    parts: List[str] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts).strip()


def run() -> None:
    random_state = SEED  # kept explicit for reproducibility
    _ = random_state

    client = make_client()

    prompts_df = pd.read_parquet(PROMPTS_PATH)
    questions_df = pd.read_parquet(QUESTIONS_PATH)

    requests, meta = build_requests(prompts_df, questions_df)
    ensure_csv(OUT_CSV)

    for chunk_i, chunk in enumerate(chunked(requests, CHUNK_SIZE), start=1):
        print(f"[INFO] Submitting batch chunk {chunk_i} with {len(chunk)} requests...")
        batch = client.messages.batches.create(requests=chunk)
        batch_id = batch.id
        print(f"[INFO] Batch created: {batch_id}")

        while True:
            b = client.messages.batches.retrieve(batch_id)
            if b.processing_status in ("ended", "completed"):
                break
            print(f"[INFO] Batch {batch_id} status: {b.processing_status} ...")
            time.sleep(POLL_SECONDS)

        results = list(client.messages.batches.results(batch_id))
        print(f"[INFO] Got {len(results)} results for batch {batch_id}")

        rows: List[dict] = []
        for r in results:
            cid = r.custom_id
            base = meta.get(cid, {}).copy()

            base.update(
                {
                    "run_id": str(uuid.uuid4()),
                    "custom_id": cid,
                    "batch_id": batch_id,
                    "judge_label": None,
                }
            )

            if r.result.type == "succeeded":
                msg = r.result.message
                base["response_text"] = extract_text_from_message(msg)

                usage = getattr(msg, "usage", None)
                base["input_tokens"] = getattr(usage, "input_tokens", None) if usage else None
                base["output_tokens"] = getattr(usage, "output_tokens", None) if usage else None
                base["error"] = None
            else:
                base["response_text"] = ""
                base["input_tokens"] = None
                base["output_tokens"] = None
                base["error"] = str(r.result)

            rows.append(base)

        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", header=False, index=False, encoding="utf-8")
        print(f"[INFO] Appended {len(rows)} rows to {OUT_CSV}")

    print(f"[DONE] All batches complete. Output: {OUT_CSV}")


if __name__ == "__main__":
    run()
