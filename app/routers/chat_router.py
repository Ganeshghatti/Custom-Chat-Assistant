from flask import Blueprint, request, jsonify
from app.controllers.chat_controller import process_chat

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint to process chat messages"""
    data = request.json
    print(f"Chat endpoint called with data: {data}")
    
    response = process_chat(data)
    return response

@chat_bp.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return "Multi-Company Support Bot API is running. Use the /chat endpoint to interact." 