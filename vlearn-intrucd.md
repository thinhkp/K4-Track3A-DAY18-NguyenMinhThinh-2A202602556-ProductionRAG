Chuẩn bị môi trường và chạy baseline
Về bài lab này
Hướng dẫn thực hành xây dựng hệ thống Production RAG gồm 5 module, đánh giá tự động bằng RAGAS và phân tích lỗi trên dữ liệu quy chế tiếng Việt.
Lab 18 — Xây dựng Pipeline Production RAG Đa Tầng
Chào mừng bạn đến với bài lab thực hành xây dựng hệ thống Production RAG. Ở bài học trước, bạn đã làm quen với Naive RAG cơ bản. Tuy nhiên, hệ thống cơ bản thường gặp nhiều vấn đề khi chạy thực tế: cắt đứt câu giữa chừng, tìm trượt các từ khóa quan trọng và dễ sinh ảo giác khi câu hỏi phức tạp.

Trong bài lab này, bạn sẽ nâng cấp hệ thống lên chuẩn Production thông qua 5 module kỹ thuật: cắt đoạn thông minh (Chunking), làm giàu thông tin (Enrichment), tìm kiếm lai (Hybrid Search), xếp hạng lại tài liệu (Reranking) và đánh giá tự động (RAGAS Evaluation).

Dưới đây là sơ đồ luồng dữ liệu của pipeline hoàn chỉnh khi vận hành:

Tài liệu gốc (data/)

M1: Cắt đoạn

M5: Làm giàu dữ liệu

M2: Tìm kiếm lai

M3: Xếp hạng lại

LLM Synthesis

M4: Đánh giá RAGAS

Báo cáo ragas_report.json

Lưu ý về sơ đồ kiến trúc và thứ tự làm bài
Sơ đồ trên mô tả luồng chạy dữ liệu của hệ thống hoàn chỉnh (Runtime Data Flow): Trong pipeline thực tế, dữ liệu sau khi cắt (M1) cần được làm giàu bằng AI (M5) trước khi nạp vào vector database và BM25 (M2).
Thứ tự thực hiện bài lab của bạn vẫn lần lượt từ M1 → M2 → M3 → M4 → M5:
Bạn code M1, M2, M3, M4 trước để hoàn thiện khung tìm kiếm cơ bản. Lúc này, pipeline trong src/pipeline.py có sẵn cơ chế tự động dùng văn bản thô (raw chunks) khi M5 chưa được cài đặt.
Sau khi có điểm số cơ bản, bạn code M5 ở bước cuối để tối ưu hóa. Khi M5 hoàn thành, pipeline sẽ tự động kích hoạt bước làm giàu dữ liệu trước M2 để nâng cao chất lượng tìm kiếm.
Toàn bộ bài tập được thực hiện trên kho quy chế nội bộ tiếng Việt gồm 28 văn bản và 20 câu hỏi đánh giá thực tế. Bạn sẽ hoàn thành bài làm cá nhân, vượt qua toàn bộ unit tests, đo lường điểm số thực tế và phân tích các trường hợp trả lời sai.

Kho mã nguồn bài tập: https://github.com/VinUni-AI20k/K4-Track3A-Day18-Production-RAG

Bạn làm được gì sau bài này
Triển khai được 3 kỹ thuật cắt đoạn: Semantic Chunking, Hierarchical Chunking (Parent-Child) và Structure-Aware Chunking.
Xây dựng cơ chế tìm kiếm lai Hybrid Search kết hợp BM25 tiếng Việt (underthesea) và Dense Search (Qdrant + bge-m3) qua thuật toán RRF.
Tích hợp tầng Cross-Encoder Reranking (bge-reranker-v2-m3) để lọc ra top-3 đoạn trích chính xác nhất.
Chấm điểm tự động pipeline với bộ 4 chỉ số RAGAS và chẩn đoán nguyên nhân lỗi bằng cây phân loại (Diagnostic Tree).
Áp dụng kỹ thuật làm giàu văn bản (Enrichment) kết hợp tóm tắt và sinh câu hỏi giả thuyết (HyQA) trước khi lưu trữ.
Cần chuẩn bị
Hiểu nguyên lý cơ bản của RAG (Naive RAG: cắt đoạn theo độ dài cố định và tìm kiếm vector đơn lẻ).
Môi trường máy tính cài đặt sẵn Python 3.11+ để hỗ trợ thư viện RAGAS và xử lý bất đồng bộ.
Đã cài đặt Docker Desktop và khởi chạy dịch vụ Qdrant cục bộ.
Có sẵn OpenAI API Key để chạy đánh giá RAGAS và làm giàu dữ liệu.
Python 3.11+
Docker & Docker Compose
Git
pytest
Lỗi thường gặp
Lỗi không kết nối được Qdrant (ConnectionRefusedError) → Chạy lệnh 'docker compose up -d' và kiểm tra cổng 6333 bằng 'docker ps'.
BM25 không tìm thấy từ khóa tiếng Việt → Thư viện underthesea nối từ ghép bằng dấu gạch dưới '_', cần đổi thành khoảng trắng bằng replace('_', ' ').
Lỗi crash thư viện FlagEmbedding → Dùng lớp CrossEncoder từ gói sentence_transformers thay vì FlagEmbedding.
Lỗi gọi hàm search() trên qdrant-client mới → Dùng phương thức query_points() thay cho search() trên qdrant-client phiên bản 1.9 trở lên.
RAGAS báo lỗi thiếu API key → Tạo file .env từ file mẫu .env.example và điền OPENAI_API_KEY hợp lệ.
Mục tiêu của phần này là khởi động dịch vụ cơ sở dữ liệu vector Qdrant, cài đặt các gói thư viện cần thiết và chạy thử hệ thống cơ sở (Naive Baseline). Điểm số của hệ thống cơ sở sẽ được dùng làm mốc so sánh trực tiếp với pipeline hoàn chỉnh mà bạn xây dựng ở các bước tiếp theo.

Đầu tiên, bạn clone repository bài lab về máy tính cá nhân. Thư mục dự án đã có sẵn dữ liệu mẫu tại data/, bộ câu hỏi kiểm thử tại test_set.json và mã nguồn khung tại src/. Bạn mở terminal và khởi tạo môi trường ảo Python theo hướng dẫn dưới đây.

Khởi tạo môi trường ảo
Linux / macOS / Git Bash
Windows PowerShell
git clone https://github.com/VinUni-AI20k/K4-Track3A-Day18-Production-RAG.git
cd K4-Track3A-Day18-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
Chép
Tiếp theo, bạn khởi động cơ sở dữ liệu vector Qdrant bằng Docker Compose, cài đặt toàn bộ thư viện phụ thuộc và tạo file cấu hình môi trường .env. File .env chứa khóa OPENAI_API_KEY dùng cho các module đánh giá và làm giàu dữ liệu bằng AI.

Cài đặt thư viện và cấu hình
Linux / macOS / Git Bash
Windows PowerShell
docker compose up -d
pip install -r requirements.txt
cp .env.example .env
Chép
Tải trước mô hình máy học
Các mô hình AI dùng trong bài có dung lượng từ vài trăm MB đến hơn 1GB. Bạn nên chạy ngay 3 lệnh dưới đây để tải trước mô hình về máy, tránh bị nghẽn mạng hoặc quá thời gian khi chấm điểm tự động.

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
Chép
Sau khi hoàn tất cài đặt, bạn chạy file naive_baseline.py. Chương trình này thực hiện tìm kiếm cơ bản: cắt đoạn thô theo dấu xuống dòng (\n\n), tìm kiếm bằng vector đơn thuần trên Qdrant và trả lời 20 câu hỏi thử nghiệm.

python naive_baseline.py
Chép
Dấu hiệu hoàn thành: Màn hình terminal in ra thông báo hoàn tất và sinh ra file kết quả tại reports/naive_baseline_report.json. File này lưu điểm số ban đầu của hệ thống để chúng ta so sánh ở bước cuối cùng.

Kiểm tra mức độ sẵn sàng0/4

Lệnh `docker ps` hiển thị container Qdrant đang chạy tại cổng 6333.

File `.env` đã được tạo và đã điền khóa `OPENAI_API_KEY`.

Cả 3 mô hình máy học đã được tải thành công về máy.

File `reports/naive_baseline_report.json` đã xuất hiện trong thư mục dự án.

Module 1 — Triển khai ba chiến lược Advanced Chunking
Ở hệ thống RAG cơ bản, văn bản thường bị cắt máy móc theo số lượng ký tự hoặc ngắt dòng đơn thuần. Cách làm này khiến câu văn dễ bị đứt giữa chừng, mất ngữ cảnh của đoạn hoặc làm vỡ cấu trúc bảng biểu. Kết quả là hệ thống tìm kiếm trích xuất ra những đoạn văn cụt lủn, thiếu ý.

Module này giúp bạn giải quyết vấn đề trên bằng 3 kỹ thuật cắt đoạn thông minh:

Semantic Chunking (Cắt theo ngữ nghĩa): Đo độ tương đồng ngữ nghĩa giữa các câu liên tiếp. Nếu câu sau vẫn cùng chủ đề với câu trước (độ tương đồng cao), hai câu được gom chung. Khi mức tương đồng giảm xuống dưới ngưỡng quy định (0.85 trong config.py), hệ thống nhận biết người viết vừa chuyển ý và ngắt sang đoạn mới.
Hierarchical Chunking (Cắt phân cấp Cha - Con): Chia văn bản thành các đoạn con nhỏ (256 ký tự) và các đoạn cha lớn bao quanh (2048 ký tự). Khi tìm kiếm, hệ thống so khớp trên đoạn con để bắt đúng từ khóa, nhưng khi gửi cho AI trả lời thì gửi cả đoạn cha để đảm bảo đầy đủ bối cảnh. Đây là cấu trúc được khuyến nghị mặc định cho các hệ thống Production RAG.
Structure-Aware Chunking (Cắt theo cấu trúc Markdown): Cắt nhỏ văn bản dựa vào các tiêu đề #, ##, ###. Cách này giúp giữ trọn vẹn từng phần quy định, không làm tách rời các bảng số liệu hay danh sách gạch đầu dòng khỏi tiêu đề quản lý của nó.
Trong thực tế doanh nghiệp, việc lựa chọn chiến lược phụ thuộc vào loại tài liệu: tài liệu có cấu trúc văn bản rõ ràng (như chính sách, luật, sổ tay) rất phù hợp với Structure-Aware và Hierarchical, trong khi các bài viết tự do hoặc báo cáo dạng tự sự lại phát huy tối đa hiệu quả với Semantic Chunking.

Các bước thực hiện trong code:

1. Mở file src/m1_chunking.py và tìm hàm chunk_semantic(). Dùng biểu thức chính quy re.split(r'(?<=[.!?])\s+|\n\n', text) tách văn bản thành danh sách câu hoàn chỉnh, dùng mô hình all-MiniLM-L6-v2 để mã hóa từng câu thành vector, sau đó tính độ tương đồng cosine giữa các câu liên tiếp để gom nhóm.
2. Tìm hàm chunk_hierarchical(). Chia văn bản thành các đoạn cha dài tối đa 2048 ký tự và gán mã parent_id (ví dụ parent_0). Sau đó chia mỗi đoạn cha thành các đoạn con tối đa 256 ký tự, nhớ lưu kèm mã parent_id vào metadata của đoạn con.
3. Tìm hàm chunk_structure_aware(). Dùng regex nhận diện các tiêu đề Markdown (#, ##), tách nội dung theo từng tiêu đề và đưa tên tiêu đề vào metadata section.
Sau khi sửa xong, bạn kiểm tra kết quả bằng lệnh pytest:

pytest tests/test_m1.py -v
Chép
Lưu ý về định dạng trả về
Hàm chunk_hierarchical() bắt buộc trả về một cặp danh sách (parents, children). Mỗi đối tượng Chunk trong danh sách con phải có thuộc tính parent_id trùng khớp với parent_id của đoạn cha tương ứng.

Dấu hiệu hoàn thành: Terminal hiển thị 100% bài kiểm tra trong test_m1.py đạt trạng thái passed. Bạn đã có cấu trúc dữ liệu cắt đoạn chuẩn xác cho các bước tiếp theo.


Module 2 — Xây dựng Hybrid Search với BM25 và Dense Search
Tìm kiếm chỉ bằng vector (Dense Search) rất giỏi hiểu ý nghĩa câu hỏi nhưng lại hay tìm trượt từ khóa chính xác. Với các câu hỏi chứa số hiệu văn bản, số ngày cụ thể hay thuật ngữ viết tắt, mô hình vector thường chỉ tìm ra các đoạn có nội dung chung chung. Ngược lại, thuật toán tìm theo tần suất từ BM25 bắt từ khóa rất chuẩn nhưng lại không hiểu được từ đồng nghĩa. Hybrid Search kết hợp cả hai để khắc phục nhược điểm của nhau.

Khi làm việc với tiếng Việt, bạn cần chú ý một điểm kỹ thuật quan trọng về tách từ. Tiếng Việt có từ ghép gồm nhiều âm tiết (ví dụ: "nghỉ phép"). Thư viện underthesea khi tách từ ghép thường nối chúng bằng dấu gạch dưới (thành nghỉ_phép). Nhưng khi người dùng gõ câu hỏi tìm kiếm, họ lại dùng khoảng trắng bình thường. Nếu giữ nguyên dấu gạch dưới, BM25 sẽ xem đây là hai từ khác nhau và tìm trượt. Vì vậy, sau khi tách từ bằng underthesea, bạn cần đổi dấu _ thành khoảng trắng.

Sau khi có kết quả từ cả BM25 và Dense Search, ta gộp thứ hạng lại bằng thuật toán RRF (Reciprocal Rank Fusion). Điểm số của mỗi đoạn văn được tính bằng công thức:

RRF_Score(d) = Σ [ 1 / (k + rank + 1) ]
Chép
Trong đó:

k là hằng số làm mượt (mặc định bằng 60 trong config.py).
rank là thứ tự xếp hạng của tài liệu trong danh sách kết quả của từng bộ tìm kiếm (bắt đầu từ 0).
Ký hiệu Σ biểu thị việc cộng dồn điểm số từ cả hai bảng kết quả (BM25 và Dense).
Tài liệu nào nằm ở thứ hạng cao ở cả hai bên sẽ nhận được điểm tổng hợp cao nhất.

Ưu điểm nổi bật của thuật toán RRF là tính độc lập với thang đo điểm số: bạn không cần phải chuẩn hóa điểm số tương đồng cosine của vector hay điểm tần suất BM25 về cùng một dải giá trị, mà chỉ cần dựa trên thứ hạng sắp xếp thực tế của từng tài liệu.

Các bước thực hiện trong code:

1. Mở file src/m2_search.py và sửa hàm segment_vietnamese(). Gọi hàm word_tokenize(text, format="text") của underthesea, sau đó dùng .replace("_", " ") để đưa về định dạng chuẩn.
2. Triển khai lớp BM25Search: trong phương thức index(), duyệt qua các đoạn văn, tách từ và đưa vào BM25Okapi. Trong phương thức search(), tách từ câu truy vấn, lấy điểm và chỉ trả về kết quả có điểm lớn hơn 0 với nhãn method="bm25".
3. Triển khai lớp DenseSearch: dùng mô hình BAAI/bge-m3 để tạo vector 1024 chiều. Phương thức index() tạo collection và lưu vector vào Qdrant. Phương thức search() mã hóa câu hỏi và gọi query_points() của Qdrant để lấy top kết quả gần nhất với method="dense".
4. Triển khai hàm reciprocal_rank_fusion(): duyệt qua hai danh sách kết quả, cộng dồn điểm RRF cho từng đoạn văn, sắp xếp giảm dần và trả về top kết quả với nhãn method="hybrid".
Chạy lệnh kiểm tra:

pytest tests/test_m2.py -v
Chép
Chú ý phiên bản qdrant-client
Từ phiên bản qdrant-client 1.9 trở lên, phương thức search() cũ đã bị thay thế bởi query_points(). Bạn hãy dùng cú pháp self.client.query_points(collection, query=query_vector, limit=top_k) để tránh bị lỗi cú pháp.

Dấu hiệu hoàn thành: Tất cả bài kiểm tra trong test_m2.py đều passed, xác nhận bộ tìm kiếm lai hoạt động ổn định và gộp kết quả chính xác từ cả hai nguồn.

Module 3 — Tối ưu độ chính xác với Cross-Encoder Reranking
Sau bước tìm kiếm lai ở Module 2, hệ thống đã lọc từ hàng trăm đoạn văn xuống còn khoảng 20 đoạn văn tiềm năng. Tuy nhiên, các bộ tìm kiếm ở tầng trước sử dụng kiến trúc Bi-Encoder: câu hỏi và văn bản được mã hóa độc lập thành hai vector rồi so sánh khoảng cách với nhau. Cách này chạy rất nhanh nhưng mô hình không thể so sánh chi tiết từng từ trong câu hỏi với từng từ trong văn bản, dẫn đến việc xếp lẫn lộn các đoạn văn tương tự nhau.

Để chọn ra đúng 3 đoạn trích đắt giá nhất cho mô hình AI đọc, chúng ta cần thêm tầng lọc thứ hai là Cross-Encoder Reranker. Khác với Bi-Encoder, Cross-Encoder nhận cùng lúc cả cặp (câu hỏi, đoạn văn) vào mô hình Transformer. Nhờ đó, mô hình có thể đối chiếu trực tiếp từng từ với nhau, nhận diện chính xác các điều khoản loại trừ hay phân biệt quy định cũ và mới.

Mặc dù Cross-Encoder tốn nhiều tài nguyên tính toán hơn, nhưng vì nó chỉ cần chấm điểm lại cho 20 ứng viên đã được sàng lọc sẵn thay vì toàn bộ kho dữ liệu hàng nghìn tài liệu, thời gian phản hồi thực tế vẫn rất nhanh và hoàn toàn đáp ứng tốt tiêu chuẩn ứng dụng thực tế.

Việc cắt lấy đúng 3 đoạn trích xuất sắc nhất (top_k=3) đóng vai trò rất quan trọng: nó giúp ngữ cảnh gửi vào mô hình tạo câu trả lời (LLM) vừa đủ ngắn gọn, tránh hiện tượng mô hình bị phân tâm bởi các thông tin ngoài lề, đồng thời tiết kiệm đáng kể chi phí token khi gọi API.

Trong bài tập này, chúng ta sử dụng mô hình BAAI/bge-reranker-v2-m3 có khả năng đọc hiểu tiếng Việt rất tốt. Với các ứng dụng đòi hỏi độ trễ cực thấp (dưới 5ms), khung mã nguồn cũng cung cấp sẵn lớp FlashrankReranker dựa trên ONNX để bạn có thể tham khảo thêm.

Các bước thực hiện trong code:

1. Mở file src/m3_rerank.py và tìm đến lớp CrossEncoderReranker.
2. Trong hàm _load_model(), import lớp CrossEncoder từ thư viện sentence_transformers và nạp mô hình BAAI/bge-reranker-v2-m3. Lưu ý không dùng thư viện FlagEmbedding vì dễ gây lỗi xung đột tokenizer trên các phiên bản mới.
3. Trong hàm rerank(), nếu danh sách văn bản rỗng thì trả về rỗng. Ngược lại, tạo danh sách các cặp (query, doc["text"]), truyền vào model.predict() để lấy điểm số liên quan cho từng đoạn.
4. Ghép điểm số với tài liệu, sắp xếp theo thứ tự điểm giảm dần, lấy ra 3 kết quả cao nhất (top_k=3) và đóng gói thành danh sách đối tượng RerankResult kèm số thứ tự rank.
Kiểm tra bằng lệnh:

pytest tests/test_m3.py -v
Chép
Đo thời gian phản hồi (Latency Benchmark)
File src/m3_rerank.py có sẵn hàm benchmark_reranker() để đo tốc độ chạy. Khi triển khai thực tế, bạn nên duy trì độ trễ của bước rerank dưới 150ms để người dùng không phải chờ đợi lâu.

Dấu hiệu hoàn thành: Lệnh test chạy thành công, xác nhận đoạn văn chứa đúng câu trả lời luôn được đẩy lên vị trí đầu bảng (rank=0).

Module 4 — Đánh giá tự động với RAGAS và cây chẩn đoán lỗi
Sau khi đã hoàn thiện các tầng tìm kiếm và xếp hạng, bạn cần một công cụ đo lường tự động để đánh giá xem câu trả lời của mô hình có thực sự đáng tin cậy hay không. Thay vì đọc và chấm điểm thủ công từng câu trả lời, chúng ta sử dụng khung đánh giá RAGAS.

Thư viện RAGAS cung cấp bộ 4 chỉ số đo lường toàn diện:

Faithfulness (Độ trung thực): Kiểm tra câu trả lời có dựa hoàn toàn vào tài liệu trích xuất hay không. Điểm thấp phản ánh hiện tượng mô hình đang tự bịa đặt thông tin ngoài tài liệu.
Answer Relevancy (Độ liên quan): Đánh giá câu trả lời có tập trung giải quyết đúng câu hỏi của người dùng hay không, tránh tình trạng trả lời lan man, lạc đề.
Context Precision (Độ chính xác ngữ cảnh): Kiểm tra xem những đoạn văn chứa đáp án đúng có được hệ thống xếp ở các vị trí đầu tiên hay không.
Context Recall (Độ bao phủ ngữ cảnh): Kiểm tra xem các đoạn văn trích xuất được có bao hàm đủ thông tin để trả lời trọn vẹn câu hỏi như đáp án chuẩn hay không.
Khi phát hiện câu hỏi có điểm số thấp, cây chẩn đoán lỗi (Diagnostic Tree) sẽ giúp bạn xác định ngay nguyên nhân nằm ở khâu nào để đưa ra hướng khắc phục:

Điểm thấp nhất ở chỉ số	Nguyên nhân chẩn đoán	Hướng xử lý đề xuất
faithfulness	LLM tự bịa câu trả lời ngoài tài liệu	Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0
context_recall	Hệ thống tìm kiếm bỏ sót đoạn văn đúng	Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25
context_precision	Đoạn văn không liên quan bị xếp lên đầu	Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata
answer_relevancy	Câu trả lời bị lệch trọng tâm câu hỏi	Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn
Các bước thực hiện trong code:

1. Mở file src/m4_eval.py. Trong hàm evaluate_ragas(), bọc code trong khối try...except. Nạp dữ liệu vào đối tượng Dataset và gọi evaluate() của RAGAS với 4 metric trên. Chuyển kết quả sang danh sách EvalResult cho từng câu hỏi.
2. Trong hàm failure_analysis(), tính điểm trung bình cho từng câu hỏi, tìm chỉ số thấp nhất (worst_metric), tra bảng trên để gán chẩn đoán (diagnosis) và gợi ý sửa (suggested_fix), sau đó sắp xếp tăng dần và trả về danh sách các câu hỏi có kết quả kém nhất.
Chạy kiểm thử:

pytest tests/test_m4.py -v
Chép
Dấu hiệu hoàn thành: Toàn bộ kiểm thử của test_m4.py đều pass, module đánh giá đã sẵn sàng tính toán điểm số cho pipeline.



Module 5 — Tối ưu nâng cao với kỹ thuật làm giàu văn bản (Enrichment)
Hiện tại, pipeline của bạn đã có đầy đủ bộ tìm kiếm, xếp hạng và đánh giá. Ở bước này, bạn sẽ bổ sung một kỹ thuật tối ưu hóa nâng cao: làm giàu văn bản (Enrichment) trước khi đưa vào cơ sở dữ liệu.

Các đoạn văn bản sau khi cắt nhỏ thường bị cụt mất tiêu đề của tài liệu gốc, khiến công cụ tìm kiếm dễ hiểu nhầm chủ đề. Kỹ thuật Contextual Prepend của Anthropic giải quyết việc này bằng cách dùng LLM viết một câu ngắn tóm tắt vị trí và chủ đề của đoạn văn trong toàn bộ tài liệu, rồi gắn trực tiếp vào đầu đoạn văn trước khi tạo vector nhúng. Bên cạnh đó, kỹ thuật sinh câu hỏi giả định (HyQA) tạo ra các câu hỏi mẫu mà đoạn văn có thể giải đáp, giúp câu truy vấn của người dùng dễ khớp trúng tài liệu hơn.

Để tối ưu chi phí và tốc độ trong môi trường thực tế, bạn nên triển khai hàm kết hợp _enrich_single_call(). Hàm này gửi một prompt duy nhất để AI cùng lúc thực hiện: tóm tắt, sinh câu hỏi mẫu, viết câu bối cảnh và trích xuất siêu dữ liệu, thay vì phải tốn 4 lần gọi API riêng biệt cho mỗi đoạn văn.

(Khi bạn hoàn thành module này, file điều phối src/pipeline.py sẽ tự động kích hoạt M5 ở khâu tiền xử lý để làm giàu toàn bộ các chunk ngay sau khi cắt ở M1 và trước khi nạp vào M2).

Các bước thực hiện trong code:

1. Mở file src/m5_enrichment.py.
2. Triển khai hàm _enrich_single_call() gửi prompt yêu cầu gpt-4o-mini phân tích đoạn văn và trả về cấu trúc JSON gồm 4 trường: summary, questions, context, metadata.
3. Hoàn thiện các hàm dự phòng (fallback) tương ứng để đảm bảo chương trình vẫn chạy được bình thường khi hệ thống không có API key.
Chạy kiểm thử:

pytest tests/test_m5.py -v
Chép
Dấu hiệu hoàn thành: Lệnh test trong test_m5.py báo passed 100%. Cả 5 module kỹ thuật của bạn đều đã sẵn sàng ghép nối vào pipeline chính thức.


Chạy tích hợp toàn bộ Pipeline và phân tích lỗi
Đây là bước bạn ghép nối toàn bộ 5 module lại với nhau và chạy thử nghiệm từ đầu đến cuối thông qua file main.py. Pipeline sẽ tự động thực hiện: cắt đoạn phân cấp (M1), làm giàu văn bản bằng AI (M5), đánh chỉ mục vào BM25 và Qdrant (M2), xếp hạng lại bằng Cross-Encoder (M3) và chấm điểm toàn diện bằng RAGAS (M4).

Bộ câu hỏi kiểm thử test_set.json gồm 20 câu thuộc nhiều dạng thực tế: câu hỏi tra cứu thông thường, câu hỏi có số liệu, câu hỏi nhiều bước và đặc biệt là câu hỏi xung đột phiên bản (giữa quy chế cũ năm 2023 và quy chế mới năm 2024).

Xung đột phiên bản là thử thách rất phổ biến trong doanh nghiệp: nếu hệ thống tìm kiếm trích xuất nhầm tài liệu cũ đã hết hiệu lực, mô hình ngôn ngữ sẽ trả lời sai lệch dù câu chữ rất mạch lạc. Sau khi chạy xong, bạn sẽ đối chiếu điểm số của Production RAG với Naive Baseline ban đầu để thấy rõ mức độ cải thiện của hệ thống mới.

Các bước thực hiện:

1. Chạy chương trình chính trên terminal:
python main.py
Chép
1. Quan sát bảng so sánh điểm số được in ra trên màn hình. Mục tiêu là các chỉ số chính (đặc biệt là Faithfulness và Context Precision) đạt từ 0.70 đến 0.75 trở lên hoặc tăng trưởng rõ rệt so với bản cũ:
============================================================
Metric                         Basic   Production         Δ
-------------------------------------------------------
✓ faithfulness                0.6200       0.8850   +0.2650
✓ answer_relevancy            0.7100       0.8400   +0.1300
✓ context_precision           0.5400       0.8100   +0.2700
✓ context_recall              0.6000       0.7900   +0.1900
Chép
1. Mở file báo cáo reports/ragas_report.json vừa được tạo ra để xem điểm số chi tiết của từng câu hỏi và danh sách các câu bị điểm thấp nhất.
2. Mở file analysis/failure_analysis.md và điền phân tích cho 5 câu hỏi bị lỗi nặng nhất (Bottom-5 Failures). Với mỗi câu, bạn trả lời 4 câu hỏi:
Câu trả lời của mô hình có đúng không?
Các đoạn trích dẫn được đưa vào có chứa đáp án không?
Câu hỏi có cần viết lại cho rõ ràng hơn không?
Cần sửa lỗi ở module nào trong pipeline?
1. Sao chép file mẫu analysis/reflections/reflection_TEMPLATE.md thành analysis/reflections/reflection_[HoVaTen].md (ví dụ reflection_NguyenVanAn.md) và hoàn thành 3 phần nội dung:
Phần 1 (Lecture Mapping): Nêu rõ từng khái niệm trên lớp tương ứng với hàm nào bạn đã viết trong code.
Phần 2 (Challenges & Debugging): Ghi lại lỗi kỹ thuật bạn gặp phải trong lúc làm bài và cách bạn đã xử lý nó.
Phần 3 (Action Plan): Lên kế hoạch áp dụng các kỹ thuật RAG này vào đồ án hoặc dự án riêng của bạn.
Dấu hiệu hoàn thành: File reports/ragas_report.json có đầy đủ số liệu, file failure_analysis.md đã điền đủ 5 ca phân tích và file reflection cá nhân đã được hoàn thành.

Kiểm tra hợp lệ và nộp bài
Trước khi nộp bài, bạn cần chạy script kiểm tra tự động để đảm bảo bài làm đúng định dạng quy định, tránh bị mất điểm thủ tục không đáng có. Hệ thống chấm điểm tự động sẽ quét chính xác tên file và cấu trúc JSON theo quy chuẩn chung của khóa học.

Dự án đã có sẵn file check_lab.py. File này sẽ tự động kiểm tra xem các file bắt buộc đã có chưa, báo cáo JSON có đúng cấu trúc không, còn dòng TODO nào trong code không và chạy lại toàn bộ unit tests. Khi script chạy hoàn tất mà không phát hiện lỗi nào, bạn có thể tự tin đẩy bài lên kho lưu trữ.

1. Kiểm tra xem còn dòng đánh dấu TODO nào chưa hoàn thiện không:
Kiểm tra TODO còn sót lại
Linux / macOS / Git Bash
Windows PowerShell
grep -r "# TODO" src/m*.py | wc -l
Chép
Mục tiêu bắt buộc là kết quả trả về bằng 0. Nếu vẫn còn số khác 0, hãy mở file tương ứng để hoàn thiện nốt.

1. Chạy script kiểm tra tổng thể:
python check_lab.py
Chép
Khi màn hình hiện dòng chữ 🚀 Bài lab sẵn sàng để nộp! và không có biểu tượng lỗi ❌, bài làm của bạn đã đạt yêu cầu kỹ thuật.

1. Kiểm tra cấu trúc thư mục repository cá nhân trước khi đẩy lên GitHub:
K4-Track3A-DAY18-HoVaTen-MSSV-ProductionRAG/
├── src/
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   └── pipeline.py
├── analysis/
│   ├── failure_analysis.md
│   └── reflections/
│       └── reflection_[HoVaTen].md
└── reports/
    └── ragas_report.json
Chép
Quy chuẩn đặt tên repository
Tên repository cá nhân trên GitHub phải đặt đúng theo mẫu: K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG. Trong đó Họ Tên viết liền không dấu (dạng PascalCase, ví dụ NguyenVanAn), mã số sinh viên viết hoa (ví dụ AI20K001). Ví dụ chuẩn: K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG.

1. Đặt repository ở chế độ Public, sao chép đường dẫn GitHub và nộp lên hệ thống VLearn LMS trước 23h59 ngày diễn ra bài lab (GMT+7).
Danh sách tự kiểm tra trước khi nộp0/6

Lệnh `python check_lab.py` thông báo bài lab đã sẵn sàng.

Tất cả unit tests trong `tests/` đều pass 100%.

Đã xóa hết các dòng đánh dấu `# TODO:` trong thư mục `src/`.

Đã hoàn thành phân tích 5 lỗi trong `analysis/failure_analysis.md`.

Đã viết xong bài thu hoạch cá nhân tại `analysis/reflections/reflection_[HoVaTen].md`.

Repository được đặt tên đúng chuẩn và đã để chế độ Public trên GitHub.
