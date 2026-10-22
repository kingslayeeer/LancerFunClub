# python benchmark.py --models qwen3:8b gemma3:12b --queries test_queries.jsonl
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import requests

from llm_module import LLMError, LLMGenerator, config

SYSTEM = (
    "Ты — помощник по деловой переписке на русском языке. Пиши грамотно, "
    "ясно и по существу, без воды и канцелярских штампов. Главную цель "
    "сообщения указывай в начале. Отвечай только текстом документа."
)

# пока так, потом взять словарь у участника 3
STAMPS = [
    "в настоящее время", "в целях", "в связи с вышеизложенным", "доводим до вашего сведения",
    "является", "осуществлять", "осуществление", "данный", "данная", "данное",
    "в рамках", "на сегодняшний день", "в соответствии с вышеуказанным",
    "имеет место", "производить", "вышеуказанный", "нижеследующий",
]


def load_queries(path):
    res = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                res.append(json.loads(line))
    return res


def text_metrics(text):
    words = re.findall(r"[A-Za-zА-Яа-яЁё-]+", text)
    sents = [s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()]
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    latin = re.findall(r"[A-Za-z]", text)
    stamps = sum(text.lower().count(s) for s in STAMPS)
    return {
        "chars": len(text),
        "words": len(words),
        "avg_sent_len": round(len(words) / max(len(sents), 1), 1),
        "latin_pct": round(100 * len(latin) / max(len(letters), 1), 1),
        "stamps_per_100w": round(100 * stamps / max(len(words), 1), 2),
    }


def get_memory(base_url, model):
    try:
        r = requests.get(base_url + "/api/ps", timeout=5)
        for m in r.json().get("models", []):
            if model in (m.get("name"), m.get("model")):
                return round(m.get("size", 0) / 1e9, 2), round(m.get("size_vram", 0) / 1e9, 2)
    except Exception:
        pass
    return 0, 0


def avg(lst):
    return round(statistics.mean(lst), 2) if lst else 0


def run(models, queries, out, base_url, backend):
    out.mkdir(parents=True, exist_ok=True)
    answers = []
    summary = []

    for model in models:
        print("\n===", model)
        llm = LLMGenerator(model=model, base_url=base_url, backend=backend, retries=0)

        # первый запрос = загрузка модели в память
        try:
            warm = llm.generate("Ответь одним словом: готов?", preset="short", max_tokens=10)
        except LLMError as e:
            print("  пропуск:", e)
            summary.append({"model": model, "error": str(e)})
            continue
        load_time = round(warm.seconds, 1)
        print("  загрузка:", load_time, "c")

        mem, vram = get_memory(base_url, model) if backend == "ollama" else (0, 0)

        times, speeds, metrics = [], [], []
        errors = 0
        for q in queries:
            try:
                r = llm.generate(q["prompt"], system=q.get("system", SYSTEM), preset=q.get("preset", "document"))
            except LLMError as e:
                errors += 1
                print(f"  {q['id']}: ошибка {e}")
                answers.append({"model": model, "id": q["id"], "answer": "ERROR: " + str(e)})
                continue

            m = text_metrics(r.text)
            metrics.append(m)
            times.append(r.seconds)
            speeds.append(r.tokens_per_second)
            print(f"  {q['id']}: {r.seconds:.1f} c, {r.tokens_per_second:.1f} ток/с")
            row = {"model": model, "id": q["id"], "query": q["prompt"], "answer": r.text,
                   "time_s": round(r.seconds, 1), "tokens": r.output_tokens}
            row.update(m)
            row["score_1_5"] = ""  # руками
            answers.append(row)

        summary.append({
            "model": model,
            "load_s": load_time,
            "avg_time_s": avg(times),
            "max_time_s": round(max(times), 1) if times else 0,
            "tok_per_s": avg(speeds),
            "mem_gb": mem,
            "vram_gb": vram,
            "errors": errors,
            "avg_sent_len": avg([m["avg_sent_len"] for m in metrics]),
            "latin_pct": avg([m["latin_pct"] for m in metrics]),
            "stamps_per_100w": avg([m["stamps_per_100w"] for m in metrics]),
        })

    save_csv(out / "answers.csv", answers)
    save_csv(out / "summary.csv", summary)
    save_md(out / "summary.md", summary)
    print("\nготово ->", out)


def get_keys(rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return keys


def save_csv(path, rows):
    if not rows:
        return
    # utf-8-sig чтобы excel нормально открыл
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=get_keys(rows), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def save_md(path, rows):
    if not rows:
        return
    keys = get_keys(rows)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--queries", default="test_queries.jsonl")
    p.add_argument("--out", default="results")
    p.add_argument("--base-url", default=config.BASE_URL)
    p.add_argument("--backend", default=config.BACKEND)
    a = p.parse_args()
    run(a.models, load_queries(a.queries), Path(a.out), a.base_url, a.backend)
