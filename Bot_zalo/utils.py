import requests
import hmac
import hashlib
from config import Config

def verify_signature(data, signature):
    return True

def send_message(user_id, text):
    #Gửi tin nhắn qua Zalo Bot Platform API
    import logging
    logger = logging.getLogger(__name__)
    
    # Sử dụng endpoint
    url = f"https://bot-api.zapps.me/bot{Config.BOT_TOKEN}/sendMessage"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'chat_id': user_id,
        'text': text
    }

    logger.info(f"Sending message to {user_id}: {text} via {url}")
    response = requests.post(url, json=payload, headers=headers)
    logger.info(f"Response status: {response.status_code}, body: {response.text}")
    
    if response.status_code != 200:
        try:
            logger.error(f"Lỗi gửi tin: {response.json()}")
        except Exception:
            logger.error(f"Lỗi gửi tin: {response.text}")
        return False
    return True