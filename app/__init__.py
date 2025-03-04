from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize MongoDB connection
mongo_uri = "mongodb+srv://ganeshghatti6:5IPx4NHMYuW8pg72@chatbot.ke4gb.mongodb.net/chatbot?retryWrites=true&w=majority&appName=chatbot"
mongo_client = MongoClient(mongo_uri)
db = mongo_client["chatbot"]
companies_collection = db["companies"]

# Make sure vector_stores directory exists
os.makedirs("vector_stores", exist_ok=True)

# Import routers after app is created to avoid circular imports
from app.routers import chat_router, company_router

# Register blueprints
app.register_blueprint(chat_router.chat_bp)
app.register_blueprint(company_router.company_bp) 