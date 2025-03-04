import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = "mongodb+srv://ganeshghatti6:5IPx4NHMYuW8pg72@chatbot.ke4gb.mongodb.net/chatbot?retryWrites=true&w=majority&appName=chatbot"

# Email configuration
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
# LLM configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama3-8b-8192"

# Vector store configuration
VECTOR_STORE_DIR = "vector_stores"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 1.25 