from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks
from src.llm_provider import chat_completion, llm_settings


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_texts = {parent.parent_id: parent.text for parent in parents}
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata,
                               "parent_id": child.parent_id,
                               "parent_text": parent_texts[child.parent_id]}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks, cache_path="reports/enrichment_cache.jsonl")
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = reranked if reranked else results[:3]
    contexts = list(dict.fromkeys(r.metadata.get("parent_text", r.text) for r in selected))

    if llm_settings()["api_key"] and contexts:
        try:
            context_str = "\n\n".join(contexts)
            resp = chat_completion(messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    input_path = "reports/production_eval_inputs.json"
    questions = [item["question"] for item in test_set]
    if os.path.exists(input_path):
        with open(input_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("questions") == questions:
            answers, all_contexts = saved["answers"], saved["contexts"]
            print("  Reusing saved answers and contexts.", flush=True)
        else:
            answers, all_contexts = [], []
    else:
        answers, all_contexts = [], []
    if not answers:
        for i, item in enumerate(test_set):
            answer, contexts = run_query(item["question"], search, reranker)
            answers.append(answer)
            all_contexts.append(contexts)
            print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
        os.makedirs("reports", exist_ok=True)
        with open(input_path, "w", encoding="utf-8") as handle:
            json.dump({"questions": questions, "answers": answers, "contexts": all_contexts},
                      handle, ensure_ascii=False, indent=2)
    ground_truths = [item["ground_truth"] for item in test_set]

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths,
                             checkpoint_path="reports/production_ragas_checkpoint.jsonl")
    print(f"  RAGAS status: {results.get('status', 'unknown')} ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    if results.get("status") == "evaluated":
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            s = results.get(m, 0)
            print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")
    else:
        print("  Chưa có điểm RAGAS hợp lệ.")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
