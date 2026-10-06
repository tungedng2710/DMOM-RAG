import logging
import os
import re
import sys
import threading
from typing import Optional

from flask import Flask, request
from dotenv import load_dotenv

# Ensure the project root (which contains the `tonrag` package) is importable
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Keep bot credentials local while sharing the web app's retrieval settings.
load_dotenv(os.path.join(BOT_DIR, '.env'))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
os.environ.setdefault('CHROMA_DIR', os.path.join(PROJECT_ROOT, 'data', 'chroma_dmom'))

from tonrag.rag import RAGPipeline  # noqa: E402
from config import Config
from utils import send_message

app = Flask(__name__)

handlers = []
if Config.DEBUG:
    handlers.append(logging.StreamHandler())
else:
    log_dir = os.path.join(BOT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'bot.log'))
    handlers.append(file_handler)
logging.basicConfig(level=logging.DEBUG if Config.DEBUG else logging.INFO, handlers=handlers,
                    format='%(asctime)s %(levelname)s %(name)s - %(message)s')
logger = logging.getLogger(__name__)

_rag: Optional[RAGPipeline] = None
_rag_lock = threading.Lock()


def get_rag() -> RAGPipeline:
    """Initialize and reuse the same pipeline as the web app on first use."""
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                _rag = RAGPipeline()
                logger.info("Initialized RAG pipeline for Zalo bot responses.")
    return _rag


def append_source_titles(answer: str, contexts: list) -> str:
    """Add document titles while preserving the pipeline's citation numbers."""
    if not answer:
        return answer
    cited = {int(number) for number in re.findall(r'\[(\d+)\]', answer)}
    sources = {}
    for number, context in enumerate(contexts, 1):
        if cited and number not in cited:
            continue
        metadata = context.get('metadata') or {}
        title = metadata.get('title') or metadata.get('document_title')
        if not title:
            reference = metadata.get('reference')
            if not reference:
                match = re.search(r'^reference:\s*(.+)$', context.get('document') or '', re.MULTILINE | re.IGNORECASE)
                reference = match.group(1) if match else ''
            # The dataset stores "Phần N, <title>, Bộ Y tế, <year>".
            title = re.sub(r'^Phần\s+\d+[A-Za-z]*\s*,\s*', '', reference.strip(), flags=re.IGNORECASE)
            title = re.sub(r',\s*Bộ Y tế\s*,?\s*(?:\d{4})?\s*$', '', title, flags=re.IGNORECASE)
        title = ' '.join((title or '').split())
        if title:
            sources.setdefault(title, []).append(number)
    if not sources:
        return answer
    lines = [f"{', '.join(f'[{number}]' for number in numbers)} {title}" for title, numbers in sources.items()]
    return answer.rstrip() + '\n\nNguồn tham khảo:\n' + '\n'.join(lines)


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
        raw_message_text = (
            # Bot Platform format thực tế từ log
            (data.get('message') or {}).get('text')
            or data.get('text')
            or ''
        )
        message_text = (raw_message_text or "").strip()

        if not user_id or not message_text:
            logger.info('Bỏ qua sự kiện: thiếu user_id hoặc tin nhắn trống.')
            return 'OK', 200

        logger.info(f'User ID: {user_id}, Message: {message_text}')

        try:
            result = get_rag().answer(message_text, top_k=5)
            reply = append_source_titles(result.get('answer') or '', result.get('contexts') or [])
            logger.info("Sinh phản hồi bằng RAG với %s nguồn.", len(result.get('contexts') or []))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Lỗi khi gọi RAG pipeline: %s", exc)
            reply = ''

        if not reply:
            reply = 'Xin lỗi, mình chưa trả lời được. Bạn hỏi lại giúp mình nhé!'

        if send_message(user_id, reply):
            logger.info(f'Gửi reply thành công cho user {user_id}')
        else:
            logger.error(f'Lỗi gửi reply cho user {user_id}')

        return 'OK', 200

if __name__ == '__main__':

    app.run(
        host="0.0.0.0", 
        port=7866,
        debug=True
    )
