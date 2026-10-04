"""Exercise the successful RAGAS conversion without external API calls."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import mock_open

import config
from src.m4_eval import evaluate_ragas, save_report


def test_ragas_scores_and_report_include_each_question(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-only")
    ragas = ModuleType("ragas")
    ragas.evaluate = lambda dataset, metrics: SimpleNamespace(
        to_pandas=lambda: SimpleNamespace(to_dict=lambda orient: [{
            "question": "Câu hỏi?", "answer": "15 ngày", "contexts": ["15 ngày"],
            "ground_truth": "15 ngày", "faithfulness": 1.0,
            "answer_relevancy": 0.8, "context_precision": 0.9, "context_recall": 1.0,
        }])
    )
    metrics = ModuleType("ragas.metrics")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(metrics, name, object())
    metrics.AnswerRelevancy = lambda strictness: object()
    datasets = ModuleType("datasets")
    datasets.Dataset = SimpleNamespace(from_dict=lambda values: values)
    monkeypatch.setitem(sys.modules, "ragas", ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    result = evaluate_ragas(["Câu hỏi?"], ["15 ngày"], [["15 ngày"]], ["15 ngày"])
    assert result["status"] == "evaluated"
    assert result["faithfulness"] == 1.0
    assert len(result["per_question"]) == 1

    opened = mock_open()
    monkeypatch.setattr("builtins.open", opened)
    save_report(result, [], "report.json")
    report = json.loads("".join(call.args[0] for call in opened().write.call_args_list))
    assert report["num_questions"] == 1
    assert report["num_evaluated"] == 1
    assert report["per_question"][0]["question"] == "Câu hỏi?"


def test_groq_uses_local_embeddings_and_one_completion(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-only")
    captured = {}

    def evaluate(dataset, metrics, **options):
        captured.update(options)
        captured["strictness"] = metrics[1].strictness
        return SimpleNamespace(to_pandas=lambda: SimpleNamespace(to_dict=lambda orient: [{
            "question": "Q", "answer": "A", "contexts": ["C"], "ground_truth": "GT",
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "context_precision": 0.5, "context_recall": 0.5,
        }]))

    ragas = ModuleType("ragas")
    ragas.evaluate = evaluate
    metrics = ModuleType("ragas.metrics")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(metrics, name, object())
    metrics.AnswerRelevancy = lambda strictness: SimpleNamespace(strictness=strictness)
    datasets = ModuleType("datasets")
    datasets.Dataset = SimpleNamespace(from_dict=lambda values: values)
    run_config = ModuleType("ragas.run_config")
    run_config.RunConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    torch = ModuleType("torch")
    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = lambda **kwargs: SimpleNamespace(**kwargs)
    langchain_core = ModuleType("langchain_core")
    rate_limiters = ModuleType("langchain_core.rate_limiters")
    rate_limiters.InMemoryRateLimiter = lambda **kwargs: SimpleNamespace(**kwargs)
    langchain_community = ModuleType("langchain_community")
    embeddings = ModuleType("langchain_community.embeddings")
    embeddings.HuggingFaceEmbeddings = lambda **kwargs: SimpleNamespace(**kwargs)
    for name, module in (("ragas", ragas), ("ragas.metrics", metrics),
                         ("ragas.run_config", run_config), ("torch", torch),
                         ("datasets", datasets), ("langchain_openai", langchain_openai),
                         ("langchain_core", langchain_core),
                         ("langchain_core.rate_limiters", rate_limiters),
                         ("langchain_community", langchain_community),
                         ("langchain_community.embeddings", embeddings)):
        monkeypatch.setitem(sys.modules, name, module)

    result = evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["status"] == "evaluated"
    assert result["provider"] == "groq"
    assert captured["llm"].base_url == "https://api.groq.com/openai/v1"
    assert captured["llm"].model == config.GROQ_EVAL_MODEL
    assert captured["embeddings"].model_name == config.EVAL_EMBEDDING_MODEL
    assert captured["strictness"] == 1
    assert captured["run_config"].max_workers == 1
    assert captured["raise_exceptions"] is True
