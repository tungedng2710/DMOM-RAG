import os
from dotenv import load_dotenv

load_dotenv() 

class Config:
    BOT_TOKEN = os.getenv('ZALO_BOT_TOKEN')  
    BOT_WEBHOOK_SECRET = os.getenv('ZALO_BOT_WEBHOOK_SECRET')  
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'