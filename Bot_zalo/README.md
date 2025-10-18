Cách tạo bot Zalo Bot Platform

I. Tạo Bot trên Zalo Bot Platform

 Mở Zalo → tìm "Zalo Bot Manager"
 Bấm vào "Tạo bot"
 Nhập tên bot (bắt đầu bằng "Bot")
 Lưu Bot Token từ tin nhắn bot gửi


-Tạo file .env và set key như bên dưới 

ZALO_BOT_TOKEN=your_bot_token_here

Webhook Secret (tự tạo)
ZALO_BOT_WEBHOOK_SECRET=your_secret_key_here

DEBUG=true



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