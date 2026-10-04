from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from src.llm_provider import llm_settings


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str],
                   checkpoint_path: str | None = None) -> dict:
    """Run RAGAS evaluation."""
    settings = llm_settings()
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("Evaluation inputs must have equal lengths")
    if not settings["api_key"]:
        print(f"  ⚠️  RAGAS unavailable: {settings['provider'].upper()}_API_KEY is missing")
        return {**{key: 0.0 for key in ("faithfulness", "answer_relevancy",
                                        "context_precision", "context_recall")},
                "per_question": [], "status": "not_evaluated_missing_api_key",
                "num_questions_input": len(questions), "provider": settings["provider"]}
    try:
        if settings["provider"] == "groq":
            import torch  # noqa: F401 -- load Windows DLLs before pyarrow/RAGAS
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metric_set = [faithfulness, answer_relevancy, context_precision, context_recall]
        evaluate_options = {}
        if settings["provider"] == "groq":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI
            from config import EVAL_EMBEDDING_MODEL, GROQ_EVAL_MODEL
            from ragas.run_config import RunConfig
            from langchain_core.rate_limiters import InMemoryRateLimiter

            evaluate_options["llm"] = ChatOpenAI(
                model=GROQ_EVAL_MODEL, api_key=settings["api_key"],
                base_url=settings["base_url"], temperature=0.01, max_retries=12,
                rate_limiter=InMemoryRateLimiter(
                    requests_per_second=1 / 35, max_bucket_size=1,
                ),
            )
            evaluate_options["embeddings"] = HuggingFaceEmbeddings(
                model_name=EVAL_EMBEDDING_MODEL
            )
            evaluate_options["run_config"] = RunConfig(
                max_workers=1, max_retries=12, max_wait=90, timeout=240
            )
            evaluate_options["raise_exceptions"] = True
            metric_set[1] = AnswerRelevancy(strictness=1)
        metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        cached = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        cached[record["key"]] = record["result"]
                    except (ValueError, KeyError):
                        continue
        per_question = []
        errors = []
        for i, (question, answer, context, ground_truth) in enumerate(
            zip(questions, answers, contexts, ground_truths)
        ):
            key = __import__("hashlib").sha256(json.dumps(
                [question, answer, context, ground_truth, settings["model"]],
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            if key in cached:
                row = cached[key]
                per_question.append(EvalResult(**row))
                continue
            try:
                dataset = Dataset.from_dict({"question": [question], "answer": [answer],
                                             "contexts": [context], "ground_truth": [ground_truth]})
                evaluated = evaluate(dataset, metrics=metric_set, **evaluate_options)
                row = evaluated.to_pandas().to_dict("records")[0]
                values = {name: float(row[name]) if math.isfinite(float(row[name])) else 0.0
                          for name in metrics}
                item = EvalResult(question, answer, context, ground_truth, **values)
                per_question.append(item)
                cached[key] = asdict(item)
                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                    with open(checkpoint_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": key, "result": asdict(item)},
                                                ensure_ascii=False) + "\n")
                print(f"  RAGAS checkpoint: {len(per_question)}/{len(questions)}", flush=True)
            except Exception as exc:
                errors.append(f"question {i + 1}: {type(exc).__name__}: {exc!r}")
                print(f"  ⚠️  RAGAS failed on question {i + 1}: {type(exc).__name__}: {exc!r}",
                      flush=True)
                break
        complete = len(per_question) == len(questions)
        return {**{key: sum(getattr(item, key) for item in per_question) / len(per_question)
                   if per_question and complete else 0.0 for key in metrics},
                "per_question": per_question,
                "status": "evaluated" if complete else "partial_evaluation",
                "error": "; ".join(errors) if errors else None,
                "num_questions_input": len(questions), "provider": settings["provider"]}
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        return {**{key: 0.0 for key in ("faithfulness", "answer_relevancy",
                                        "context_precision", "context_recall")},
                "per_question": [], "status": "not_evaluated_error",
                "error": f"{type(exc).__name__}: {exc!r}",
                "num_questions_input": len(questions), "provider": settings["provider"]}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "faithfulness": ("Câu trả lời thiếu căn cứ trong ngữ cảnh", "Siết prompt và giảm temperature"),
        "context_recall": ("Thiếu đoạn liên quan", "Điều chỉnh chunking và truy xuất BM25"),
        "context_precision": ("Có nhiều đoạn không liên quan", "Cải thiện reranking hoặc lọc metadata"),
        "answer_relevancy": ("Câu trả lời lệch câu hỏi", "Điều chỉnh prompt tổng hợp"),
    }
    metrics = tuple(tree)
    ranked = sorted(eval_results, key=lambda item: sum(getattr(item, m) for m in metrics) / 4)
    failures = []
    for item in ranked[:bottom_n]:
        worst = min(metrics, key=lambda m: getattr(item, m))
        diagnosis, fix = tree[worst]
        failures.append({"question": item.question, "expected": item.ground_truth,
                         "answer": item.answer, "worst_metric": worst,
                         "score": getattr(item, worst), "diagnosis": diagnosis,
                         "suggested_fix": fix})
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: (v if results.get("status", "evaluated") == "evaluated" else None)
                      for k, v in results.items()
                      if k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
        "num_questions": results.get("num_questions_input", len(results.get("per_question", []))),
        "num_evaluated": len(results.get("per_question", [])),
        "status": results.get("status", "evaluated"),
        "provider": results.get("provider"),
        "error": results.get("error"),
        "per_question": [asdict(item) if isinstance(item, EvalResult) else item
                         for item in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
