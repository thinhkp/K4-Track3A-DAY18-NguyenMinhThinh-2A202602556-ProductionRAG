# Failure Analysis — Lab 18

## Trạng thái đánh giá

Pipeline Production đã tạo câu trả lời cho đủ 20 câu hỏi. RAGAS bắt đầu chấm với Groq nhưng dừng ở 55/80 lượt metric; lần chạy không lưu được điểm từng câu, vì vậy hiện chưa có điểm tổng hợp hoặc bottom-5 hợp lệ. Baseline nhận HTTP 429 do giới hạn TPM: 5.513 tokens đã dùng cùng yêu cầu 2.657 tokens vượt giới hạn 8.000 tokens/phút. Đây là giới hạn lưu lượng theo phút, không phải bằng chứng key hết hạn mức ngày.

Evaluator hiện lưu checkpoint sau mỗi câu hỏi, dùng một worker và giãn lời gọi để giảm lỗi TPM. Chạy lại `python src/pipeline.py` khi có thể gọi API; các câu trả lời/context sẽ được lưu để lần chạy kế tiếp tái sử dụng.

Baseline chưa có kết quả RAGAS hợp lệ. Docker daemon chưa chạy nên vector search dùng Qdrant in-memory. Hai PDF scan không có lớp văn bản nên bị bỏ qua, cần OCR để đưa chúng vào pipeline.

| Metric | Naive Baseline | Production | Delta |
|---|---:|---:|---:|
| Faithfulness | Chưa đo | Chưa hoàn tất | — |
| Answer Relevancy | Chưa đo | Chưa hoàn tất | — |
| Context Precision | Chưa đo | Chưa hoàn tất | — |
| Context Recall | Chưa đo | Chưa hoàn tất | — |

## Cây chẩn đoán sau khi có kết quả

1. Câu trả lời sai và context thiếu chứng cứ: kiểm tra context recall; cải thiện chunking, BM25, phiên bản tài liệu và OCR cho PDF scan.
2. Context có chứng cứ nhưng nhiều đoạn không liên quan: kiểm tra context precision; điều chỉnh reranker và lọc metadata.
3. Context đúng nhưng câu trả lời thêm thông tin không có căn cứ: kiểm tra faithfulness; siết prompt và giảm temperature.
4. Câu trả lời không đáp ứng câu hỏi dù có căn cứ: kiểm tra answer relevancy; điều chỉnh prompt tổng hợp.

`failure_analysis()` trong `src/m4_eval.py` tự động chọn các câu điểm thấp nhất và gán nguyên nhân cùng hướng sửa. Sau khi RAGAS và baseline được chấm đủ 20 câu, cập nhật bảng so sánh và bottom-5 thực nghiệm từ các báo cáo JSON.

