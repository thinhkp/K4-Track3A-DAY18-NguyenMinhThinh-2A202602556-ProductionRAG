from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_provider import chat_completion, llm_settings


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Enrichment helpers ──────────────────────────────────

import json
import re
import hashlib
from dataclasses import asdict


def _ask(system: str, text: str, max_tokens: int = 300, json_mode: bool = False) -> str:
    if not llm_settings()["api_key"]:
        return ""
    model = llm_settings()["model"]
    options = {"messages": [{"role": "system", "content": system},
                            {"role": "user", "content": text}],
               "max_tokens": max_tokens}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    if llm_settings()["provider"] == "groq" and model.startswith("openai/gpt-oss-"):
        options["reasoning_effort"] = "low"
    response = chat_completion(**options)
    return (response.choices[0].message.content or "").strip()


def summarize_chunk(text: str) -> str:
    """Summarize a chunk, with an extractive fallback."""
    try:
        result = _ask("Tóm tắt đoạn sau trong 2 câu tiếng Việt, chỉ dùng thông tin có trong đoạn.", text, 150)
        if result:
            return result
    except Exception as exc:
        print(f"  ⚠️  Summary failed: {exc}")
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    return " ".join(sentences[:2])


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions answerable from the chunk."""
    try:
        result = _ask(f"Tạo {n_questions} câu hỏi tiếng Việt có thể trả lời từ đoạn sau. Mỗi dòng một câu hỏi.", text)
        if result:
            return [re.sub(r'^\s*\d+[.)]\s*', '', line).strip()
                    for line in result.splitlines() if line.strip()][:n_questions]
    except Exception as exc:
        print(f"  ⚠️  HyQA failed: {exc}")
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"Nội dung nào được quy định về {s[:80]}?" for s in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Add source context while preserving the full original text."""
    try:
        context = _ask("Viết một câu tiếng Việt nêu chủ đề và vị trí của đoạn trong tài liệu.",
                       f"Tài liệu: {document_title}\nĐoạn: {text}", 80)
        if context:
            return f"{context}\n\n{text}"
    except Exception as exc:
        print(f"  ⚠️  Context generation failed: {exc}")
    return f"Trích từ {document_title}.\n\n{text}" if document_title else text


def extract_metadata(text: str) -> dict:
    """Extract structured metadata, falling back to neutral values."""
    try:
        response = _ask('Trả về JSON hợp lệ gồm topic, entities (array), category, language cho đoạn văn.',
                        text, 150, json_mode=True)
        if response:
            return json.loads(response)
    except Exception as exc:
        print(f"  ⚠️  Metadata extraction failed: {exc}")
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    """Get summary, HyQA, context and metadata in one API request."""
    if not llm_settings()["api_key"]:
        return {}
    try:
        response = _ask(
            'Trả về JSON hợp lệ với keys summary (string), questions (array 3 câu hỏi), '
            'context (một câu), metadata (object gồm topic, entities, category, language). '
            'Chỉ dùng thông tin trong đoạn.',
            f"Tài liệu: {source}\n\nĐoạn: {text}", 1200, json_mode=True,
        )
        return json.loads(response)
    except Exception as exc:
        print(f"  ⚠️  Combined enrichment failed: {exc}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    cache_path: str | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    cache[row["key"]] = row["chunk"]
                except (ValueError, KeyError):
                    continue

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        cache_key = hashlib.sha256(json.dumps(
            [text, source, methods, llm_settings()["model"]], ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        if cache_key in cache:
            enriched.append(EnrichedChunk(**cache[cache_key]))
            continue

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            summary = summary if isinstance(summary, str) else ""
            raw_questions = result.get("questions", [])
            questions = [q for q in raw_questions if isinstance(q, str)][:3] if isinstance(raw_questions, list) else []
            context_line = result.get("context", "")
            context_line = context_line if isinstance(context_line, str) else ""
            enrichment = []
            if context_line:
                enrichment.append(context_line)
            if summary:
                enrichment.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment.append("Câu hỏi liên quan: " + " | ".join(questions))
            enriched_text = "\n".join(enrichment) + "\n\n" + text if enrichment else text
            auto_meta = result.get("metadata", {})
            auto_meta = auto_meta if isinstance(auto_meta, dict) else {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            enrichment = []
            if summary:
                enrichment.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment.append("Câu hỏi liên quan: " + " | ".join(questions))
            if enrichment:
                enriched_text = "\n".join(enrichment) + "\n\n" + enriched_text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        item = EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )
        enriched.append(item)
        if cache_path and (not use_combined or result):
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": cache_key, "chunk": asdict(item)},
                                        ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
