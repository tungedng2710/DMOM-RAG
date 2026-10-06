# Cách tạo bot Zalo Bot Platform

## Trả lời bằng RAG như web app

Bot sử dụng `tonrag.rag.RAGPipeline.answer()` với `top_k=5`, giống endpoint
`/api/chat` của web app. Câu hỏi được truy xuất từ Chroma, sau đó gửi cùng
ngữ cảnh đến GLM 5.3 vLLM; câu trả lời giữ nguyên trích dẫn `[1]`, `[2]`.
Cuối tin nhắn có mục `Nguồn tham khảo` chỉ liệt kê tên tài liệu được trích dẫn,
không kèm số phần, nhà xuất bản hay năm xuất bản. Các trích dẫn cùng tên tài
liệu được gộp trên một dòng. Nếu câu trả lời không có trích dẫn, bot liệt kê
tên tài liệu đã truy xuất; nguồn không có tên được bỏ qua.
Nếu LLM không hoạt động, pipeline sử dụng câu trả lời từ tài liệu truy xuất
đầu tiên, giống web app.

Cấu hình Zalo nằm trong `Bot_zalo/.env`. Bot cũng đọc `.env` ở thư mục gốc
cho `CHROMA_DIR`, `CHROMA_COLLECTION`, `OLLAMA_BASE_URL` và các biến `VLLM_*`.
Biến môi trường đã có sẵn được ưu tiên, sau đó là `Bot_zalo/.env`, rồi `.env`
ở thư mục gốc. Đường dẫn Chroma tương đối được tính từ thư mục gốc dự án.

```bash
conda activate tungn197
pip install -r Bot_zalo/requirements.txt
cd Bot_zalo
python -m flask --app app run --host 0.0.0.0 --port 7866 --no-reload
```

Cần có collection Chroma đã được tạo và dịch vụ embeddings Ollama với model
`bge-m3:latest`. URL webhook cho ngrok là `https://<ngrok-host>/webhook`.

## Tạo Bot trên Zalo Bot Platform
1. Mở Zalo → tìm "Zalo Bot Manager".
2. Bấm vào "Tạo bot".
3. Nhập tên bot (bắt đầu bằng "Bot").
4. Lưu Bot Token từ tin nhắn bot gửi.
5. Tạo file .env và set key như bên dưới.
`
ZALO_BOT_TOKEN=your_bot_token_here
`
6. Webhook Secret (tự tạo).
`
ZALO_BOT_WEBHOOK_SECRET=your_secret_key_here
DEBUG=true
VLLM_BASE_URL=http://171.232.252.135:31585/v1
VLLM_MODEL=glm53-flash
VLLM_API_KEY=your_bearer_token
`



II. Tạo tunnel công khai
# Sử dụng serveo.net
ssh -o StrictHostKeyChecking=accept-new -R testzalobot:80:localhost:5000 serveo.net

III. Thiết lập Webhook

# Test Bot Token
curl -X GET "https://bot-api.zapps.me/bot${ZALO_BOT_TOKEN}/getMe"

# Set Webhook
curl -X POST "https://bot-api.zapps.me/bot${ZALO_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "{url được serveo tạo ra}/webhook",
    "secret_token": "your_secret_key_here"
  }'

# Cần phải mở 2 terminal để chạy
1. Chạy bot : python3 app.py
2. Tạo tunnel công khai: ssh -o StrictHostKeyChecking=accept-new -R testzalobot:80:localhost:5000 serveo.net

# Link tài liệu :
https://bot.zapps.me/docs
