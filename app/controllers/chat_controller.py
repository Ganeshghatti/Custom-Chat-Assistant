from app.utils.llm_utils import chat

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
    
    response = chat(user_input, company_id, conversation_history)
    
    # Return the response
    return {"response": response}, 200 