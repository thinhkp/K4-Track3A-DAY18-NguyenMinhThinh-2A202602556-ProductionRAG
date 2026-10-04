"""Ensure generated enrichment actually reaches the search index text."""

from src import m5_enrichment


def test_combined_enrichment_indexes_summary_and_questions(monkeypatch):
    monkeypatch.setattr(
        m5_enrichment,
        "_enrich_single_call",
        lambda text, source: {
            "summary": "Quy định nghỉ phép năm.",
            "questions": ["Được nghỉ bao nhiêu ngày?"],
            "context": "Quy chế nhân sự hiện hành.",
            "metadata": {"topic": "nghỉ phép"},
        },
    )
    original = "Nhân viên được nghỉ 15 ngày."
    enriched = m5_enrichment.enrich_chunks(
        [{"text": original, "metadata": {"source": "policy.md"}}]
    )[0]

    assert original in enriched.enriched_text
    assert enriched.summary in enriched.enriched_text
    assert enriched.hypothesis_questions[0] in enriched.enriched_text
    assert "Quy chế nhân sự" in enriched.enriched_text
    assert enriched.auto_metadata["source"] == "policy.md"
