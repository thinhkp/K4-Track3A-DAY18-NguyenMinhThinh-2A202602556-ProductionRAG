# Individual Reflection — Lab 18

**Họ tên:** NguyenMinhThinh  
**MSSV:** 2A202602556  
**Khóa:** K4 – Track 3A  
**Ngày:** 04/10/2026

## 1. Lecture mapping

| Khái niệm | Mã đã triển khai | Quan sát đã kiểm chứng |
|---|---|---|
| Semantic chunking | `src/m1_chunking.py:chunk_semantic()` | Tách câu, mã hóa bằng all-MiniLM-L6-v2 và ngắt khi cosine dưới ngưỡng; test M1 đạt. Chưa đo số chunk so với baseline trên toàn corpus. |
| Hierarchical và structure-aware chunking | `chunk_hierarchical()`, `chunk_structure_aware()` | Child tối đa 256 ký tự, parent tối đa 2048 ký tự; pipeline tìm child và gửi parent làm context. Header Markdown được giữ trong chunk và metadata. |
| BM25 + Dense + RRF | `src/m2_search.py:BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Tách từ tiếng Việt và thay `_` bằng khoảng trắng; tìm vector bằng Qdrant `query_points()`; RRF cộng điểm theo thứ hạng. Test M2 đạt. |
| Cross-encoder reranking | `src/m3_rerank.py:CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 chấm cặp câu hỏi–đoạn văn; trả top 3. Test M3 đạt, chưa có benchmark latency chính thức. |
| RAGAS 4 metrics | `src/m4_eval.py:evaluate_ragas()` | Groq key đã được cấu hình; Production đã tạo câu trả lời cho 20 câu. RAGAS dừng ở 55/80 metric calls sau rate limit TPM 8.000, nên chưa có điểm hợp lệ. |
| Contextual enrichment, summary, HyQA | `src/m5_enrichment.py:_enrich_single_call()` và các hàm riêng | Combined mode dùng một lời gọi API/chunk; không có key thì dùng văn bản thô và fallback cục bộ. Test M5 đạt. |

## 2. Khó khăn và cách xử lý

- Lần đầu `pytest` báo `ModuleNotFoundError: No module named 'sentence_transformers'`, cùng lỗi thiếu `underthesea` và `rank_bm25`. Tôi cài dependencies bằng Python 3.12 vào `.deps`, chạy lại và đạt 40/40 test.
- Docker báo `failed to connect to the docker API ... dockerDesktopLinuxEngine`. DenseSearch dùng Qdrant in-memory khi server không sẵn sàng; baseline và pipeline offline đã xử lý 20 câu.
- Groq trả HTTP 429 khi RAGAS vượt giới hạn 8.000 tokens/phút (một lần thử dùng 5.513 tokens và yêu cầu thêm 2.657). Đã hạ tốc độ đánh giá xuống một worker và thêm checkpoint theo câu hỏi; cần chạy lại khi API cho phép.
- Hai PDF scan không có text layer nên bị bỏ qua và cần OCR trước khi đánh giá đầy đủ corpus. BGE-M3 yêu cầu dung lượng model lớn; cần kiểm tra dung lượng ổ đĩa và cache Hugging Face khi triển khai.

## 3. Kế hoạch áp dụng

**Dự án áp dụng:** Trợ lý tra cứu quy chế nội bộ, dùng chính corpus của lab làm điểm xuất phát.

- **Tuần 1:** Chuẩn hóa tài liệu, thêm OCR cho PDF scan, gắn phiên bản và hiệu lực vào metadata; so sánh hierarchical với structure-aware chunking.
- **Tuần 2:** Dùng BM25 + dense + RRF, thêm lọc theo phiên bản hiện hành và rerank top 20 xuống top 3. Đo Recall@k và latency từng tầng.
- **Tuần 3:** Chạy baseline và production trên cùng 20 câu hỏi; đo bốn chỉ số RAGAS, điền bottom-5 và sửa theo cây chẩn đoán. Chỉ công bố so sánh sau khi cả hai báo cáo có điểm hợp lệ.

Cần chạy lại RAGAS và baseline đủ 20 câu, sau đó bổ sung điểm thực nghiệm và bottom-5 trước khi nộp đầy đủ.



