import logging

from flask import Flask, jsonify, request

from config import Config
from utils import send_message, verify_signature

app = Flask(__name__)

handlers = []
if Config.DEBUG:
    handlers.append(logging.StreamHandler())
else:
    file_handler = logging.FileHandler('logs/bot.log')
    handlers.append(file_handler)
logging.basicConfig(level=logging.DEBUG if Config.DEBUG else logging.INFO, handlers=handlers,
                    format='%(asctime)s %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Homepage"""
    return 'Zalo Bot is running! Webhook: /webhook', 200

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        # Xác minh challenge từ Zalo
        challenge = request.args.get('challenge')
        if challenge:
            return challenge
        return 'OK', 200
    
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        raw_data = request.data  
        
        # Verify Bot Platform secret token 
        if Config.BOT_WEBHOOK_SECRET:
            provided = request.headers.get('X-Bot-Api-Secret-Token')
            if not provided or provided != Config.BOT_WEBHOOK_SECRET:
                logger.error('Invalid X-Bot-Api-Secret-Token')
                return 'Invalid secret token', 403
        
        # Xử lý sự kiện
        event = data.get('event_name') or data.get('event') or 'bot_platform_message'
        logger.info(f'Nhận event: {event}')
        logger.debug(f'Full data: {data}')  
        
        # Lấy user_id và text 
        user_id = (
            # Bot Platform format thực tế từ log
            (data.get('message') or {}).get('from', {}).get('id')
            or data.get('recipient', {}).get('id')
            or data.get('sender', {}).get('id')
            or data.get('user_id')
            or data.get('chat_id')
        )
        message_text = (
            # Bot Platform format thực tế từ log
            (data.get('message') or {}).get('text')
            or data.get('text')
            or ''
        ).lower()
        if not user_id or not message_text:
            # Early exist if the user ID or message is invalid.
            return
            logger.info(f'User ID: {user_id}, Message: {message_text}')
            
            # Logic bot
            if 'xin chào' in message_text:
                reply = 'Chào bạn! Cần hỗ trợ gì?'
            elif 'tạm biệt' in message_text:
                reply = 'Tạm biệt! Hẹn gặp lại.'
            else:
                reply = 'Tôi chưa hiểu, bạn hỏi lại nhé?'
            
            # Gửi phản hồi
            if send_message(user_id, reply):
                logger.info(f'Gửi reply thành công cho user {user_id}')
            else:
                logger.error(f'Lỗi gửi reply cho user {user_id}')
        else:
            logger.info('Bỏ qua sự kiện không phải tin nhắn văn bản hoặc thiếu user_id/text')
        
        return 'OK', 200

if __name__ == '__main__':

    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=Config.DEBUG
    )