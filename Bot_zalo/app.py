import logging
import os
import sys
from typing import Optional

from flask import Flask, request

# Ensure the project root (which contains the `tonrag` package) is importable
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tonrag.llm import CerebrasChat, GeminiChat, get_default_chat  # noqa: E402
from config import Config
from utils import send_message

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

SYSTEM_PROMPT = (
    "Bạn là bác sĩ trả lời câu hỏi y khoa bằng tiếng Việt, ngắn gọn và chính xác. "
    "Chỉ sử dụng kiến thức đã được xác minh. Nếu thiếu thông tin, hãy nói: "
    '"Không đủ thông tin để trả lời đầy đủ". '
    "Luôn giữ giọng điệu chuyên nghiệp và dễ hiểu."
)

_chat_client: Optional[object] = None
_chat_backend: Optional[str] = None
_chat_initialization_error: Optional[str] = None


def _resolve_backend() -> str:
    """Determine which chat backend to use for the bot."""
    raw = (
        os.getenv("ZALO_CHAT_BACKEND")
        or os.getenv("CHAT_BACKEND")
        or "gemini"
    )
    choice = (raw or "").strip().lower()
    if choice.startswith("ollama"):
        return "ollama"
    if choice.startswith("cerebras"):
        return "cerebras"
    if choice.startswith("gemini"):
        return "gemini"
    return "gemini"


def _resolve_api_key(backend: str) -> Optional[str]:
    env_map = {
        "gemini": "GEMINI_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }
    env_name = env_map.get(backend)
    if not env_name:
        return None
    value = (os.getenv(env_name, "") or "").strip()
    return value or None


def get_chat_client() -> Optional[object]:
    """Lazy-load the chat client for reuse across requests."""
    global _chat_client, _chat_backend, _chat_initialization_error
    backend = _resolve_backend()
    needs_refresh = (_chat_client is None) or (_chat_backend != backend)
    if needs_refresh or _chat_initialization_error is not None:
        try:
            api_key_override = _resolve_api_key(backend)
            _chat_client = get_default_chat(backend, api_key=api_key_override)
            if isinstance(_chat_client, GeminiChat):
                _chat_backend = "gemini"
            elif isinstance(_chat_client, CerebrasChat):
                _chat_backend = "cerebras"
            else:
                _chat_backend = "ollama"
            _chat_initialization_error = None
            logger.info("Initialized chat backend '%s' for Zalo bot responses.", backend)
        except Exception as exc:  # pragma: no cover - defensive
            _chat_client = None
            _chat_backend = backend
            _chat_initialization_error = str(exc)
            logger.exception("Failed to initialize chat backend '%s': %s", backend, exc)
    return _chat_client


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

        chat = get_chat_client()
        backend_name = _chat_backend or _resolve_backend()
        if chat is None:
            logger.error(
                "Chat backend '%s' chưa sẵn sàng: %s",
                backend_name,
                _chat_initialization_error or "unknown error",
            )
            reply = 'Xin lỗi, hệ thống đang bận. Bạn thử lại sau nhé!'
        else:
            messages = [
                {
                    "role": "user",
                    "content": message_text,
                }
            ]
            try:
                reply = chat.generate(messages, system=SYSTEM_PROMPT) or ''
                logger.info("Sinh phản hồi thành công bằng backend '%s'.", backend_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Lỗi khi gọi backend '%s': %s", backend_name, exc)
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
        port=7872, 
        debug=True
    )
