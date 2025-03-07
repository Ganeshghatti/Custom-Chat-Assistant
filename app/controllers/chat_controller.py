from flask import Response, stream_with_context
from app.utils.llm_utils import chat
import json

def process_chat(data):
    """Process a chat request"""
    if not data or 'message' not in data:
        return {"error": "No message provided"}, 400
    
    if 'company_id' not in data:
        return {"error": "No company_id provided"}, 400
    
    user_input = data['message']
    company_id = data['company_id']
    conversation_history = data.get('conversation_history', [])
    
    # Limit to last 10 exchanges to maintain context without overwhelming
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    # Get the streaming generator
    stream_generator = chat(user_input, company_id, conversation_history)
    
    # Create a streaming response
    def generate():
        for chunk in stream_generator:
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    ) 