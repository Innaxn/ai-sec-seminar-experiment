import os
import time
import random
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import httpx
from openai import OpenAI


PROMPTS_PATH = "data/prompts.parquet"
QUESTIONS_PATH = "data/questions.parquet"
OUT_CSV = "results_pilot_gpt5mini.csv"

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

MODEL_NAME = "gpt-5-mini"

SLEEP_SECONDS = 0.2
MAX_RETRIES = 3
TIMEOUT_SECONDS = 45.0
FLUSH_EVERY_N_ROWS = 5

PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00

PRINT_RESPONSE_SNIPPET = True
SNIPPET_CHARS = 200


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_input_text(jb: Optional[str], q: str) -> str:
    q = q.strip()
    if jb is None:
        return q
    return f"{jb.strip()}\n\nUser question:\n{q}"


def estimate_cost_usd(in_tok: Optional[int], out_tok: Optional[int]) -> Optional[float]:
    if in_tok is None or out_tok is None:
        return None
    return (in_tok * PRICE_INPUT_PER_1M / 1_000_000) + (out_tok * PRICE_OUTPUT_PER_1M / 1_000_000)


def flush_rows(rows: list[dict], path: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8")
    rows.clear()


def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI(
        api_key=api_key,
        http_client=httpx.Client(timeout=TIMEOUT_SECONDS),
    )


def call_model_with_retry(
    client: OpenAI, input_text: str
) -> Tuple[str, Optional[int], Optional[int], Optional[int], Optional[str]]:
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(model=MODEL_NAME, input=input_text)
            text = getattr(resp, "output_text", "") or ""
            usage = getattr(resp, "usage", None)

            in_tok = getattr(usage, "input_tokens", None) if usage else None
            out_tok = getattr(usage, "output_tokens", None) if usage else None
            tot_tok = getattr(usage, "total_tokens", None) if usage else None

            return text, in_tok, out_tok, tot_tok, None
        except Exception as e:
            last_err = str(e)
            backoff = min(15, (2 ** (attempt - 1)) + random.random())
            print(f"[WARN] attempt {attempt}/{MAX_RETRIES} failed: {last_err}")
            print(f"[WARN] sleeping {backoff:.1f}s then retrying...")
            time.sleep(backoff)

    return "", None, None, None, f"FAILED_AFTER_RETRIES: {last_err}"


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


def run_condition(
    client: OpenAI,
    qf: pd.DataFrame,
    jb_sample: Optional[pd.DataFrame],
    rows: list[dict],
    out_csv: str,
    start_done: int,
    total_planned: int,
    condition: str,
) -> int:
    done = start_done

    for _, qrow in qf.iterrows():
        cat = str(qrow[CATEGORY_COL])
        qtext = str(qrow[QUESTION_TEXT_COL])
        qkey = str(qrow["question_key"])

        jb_iter = [(None, None, None, None)] if jb_sample is None else [
            (
                str(prow[PROMPT_TEXT_COL]),
                str(prow["jailbreak_key"]),
                str(prow[PLATFORM_COL]) if PLATFORM_COL in prow and pd.notna(prow[PLATFORM_COL]) else None,
                str(prow[SOURCE_COL]) if SOURCE_COL in prow and pd.notna(prow[SOURCE_COL]) else None,
            )
            for _, prow in jb_sample.iterrows()
        ]

        for jb_text, jbkey, platform, source in jb_iter:
            tag = "baseline" if jb_text is None else "jailbreak"
            print(f"[SEND] {tag} | cat={cat} | qkey={qkey}" + (f" | jbkey={jbkey}" if jbkey else ""))

            out, in_tok, out_tok, tot_tok, err = call_model_with_retry(
                client, build_input_text(jb_text, qtext)
            )

            est_cost = estimate_cost_usd(in_tok, out_tok)
            print(
                f"  -> output_empty={len(out.strip())==0}, tokens(in/out/total)={in_tok}/{out_tok}/{tot_tok}, cost≈{est_cost}"
            )
            if PRINT_RESPONSE_SNIPPET and out:
                print("  -> snippet:", out[:SNIPPET_CHARS].replace("\n", " "))

            rows.append(
                {
                    "run_id": str(uuid.uuid4()),
                    "timestamp_utc": utc_now_iso(),
                    "condition": condition,
                    "category": cat,
                    "question_key": qkey,
                    "question_prompt": qtext,
                    "jailbreak_key": jbkey,
                    "jailbreak_prompt": jb_text,
                    "platform": platform,
                    "source": source,
                    "model": MODEL_NAME,
                    "response_text": out,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "estimated_cost_usd": est_cost,
                    "error": err,
                    "judge_label": None,
                }
            )

            done += 1
            if done % FLUSH_EVERY_N_ROWS == 0:
                flush_rows(rows, out_csv)
                print(f"[INFO] Saved {done}/{total_planned} rows")

            time.sleep(SLEEP_SECONDS)

    return done


def main() -> None:
    random.seed(SEED)

    client = make_client()

    print("[INFO] Loading parquets...")
    prompts_df = pd.read_parquet(PROMPTS_PATH)
    questions_df = pd.read_parquet(QUESTIONS_PATH)

    qf = sample_questions(questions_df)
    jb_sample = sample_jailbreaks(prompts_df)

    print("[INFO] Sampled questions:")
    print(qf[[CATEGORY_COL, QUESTION_TEXT_COL]].head(10))
    print(f"[INFO] Sampled jailbreak prompts: {len(jb_sample)}")

    rows: list[dict] = []
    total_planned = len(qf) + (len(qf) * len(jb_sample))
    done = 0

    print("\n[INFO] BASELINE...")
    done = run_condition(
        client=client,
        qf=qf,
        jb_sample=None,
        rows=rows,
        out_csv=OUT_CSV,
        start_done=done,
        total_planned=total_planned,
        condition="baseline",
    )
    flush_rows(rows, OUT_CSV)

    print("\n[INFO] JAILBREAK...")
    done = run_condition(
        client=client,
        qf=qf,
        jb_sample=jb_sample,
        rows=rows,
        out_csv=OUT_CSV,
        start_done=done,
        total_planned=total_planned,
        condition="jailbreak",
    )
    flush_rows(rows, OUT_CSV)

    print(f"\n[DONE] Pilot complete. Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
