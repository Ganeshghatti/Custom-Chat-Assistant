from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import tool, create_react_agent, AgentExecutor
from app.utils.vector_store_utils import get_vector_store, embeddings
from app.utils.email_utils import send_email
from app.config import LLM_MODEL, SIMILARITY_THRESHOLD

# Initialize the LLM
llm = ChatGroq(model=LLM_MODEL)

@tool
def send_message_to_admin(name: str, email: str, message: str, company_id: str = None):
    """ Takes user's name, email and message and sends it to admin """
    print(f"Sending message to admin with name {name}, email {email} and message {message}")
    # Send email to admin
    success = send_email(name, email, message, company_id)
    
    print(f"Success: {success}")
    if success:
        return f"Message sent to admin with name {name}, email {email} and message {message}"
    else:
        return f"Failed to send message to admin. Please try again later."

tools = [send_message_to_admin]

def chat(user_input, company_id, conversation_history=None):
    """Process a chat message and return a response"""
    print(f"Chat function called with user input: {user_input} and company ID: {company_id}")
    if conversation_history is None:
        conversation_history = []
    
    try:
        # Get the vector store and company name
        company_data = get_vector_store(company_id)
        vector_store = company_data["store"]
        company_name = company_data["name"]
        
        # Create embedding for the user query
        query_embedding = embeddings.embed_query(user_input)
        
        # Retrieve relevant context based on the user query with similarity scores
        docs = vector_store.similarity_search_with_score(user_input, k=3)
        
        # Check if we have any relevant documents (using a threshold)
        print(f"docs {docs}")
        print(f"similarity_threshold {SIMILARITY_THRESHOLD}")
        
        relevant_docs = [doc for doc, score in docs if score <= SIMILARITY_THRESHOLD]
    
        # Extract context from relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Format conversation history for context
        history_text = ""
        if conversation_history:
            for msg in conversation_history:
                role = "Assistant" if msg['role'] == 'assistant' else "User"
                history_text += f"{role}: {msg['content']}\n"
        
        # Create a dynamic system message based on company
        system_message = f"""You are a specialized support assistant for {company_name}.
        
        IMPORTANT GUIDELINES:
        1. ONLY answer questions related to {company_name}, its products, services, or company information
        2. If asked about unrelated topics, politely redirect the conversation to {company_name}
        3. Use the provided context and conversation history to give accurate, helpful responses
        4. Be concise and professional in your responses
        5. If the user wants to contact an admin or support team, use the send_message_to_admin tool. if arguments needed to send email aren't provided, then ask user to provide them.
        
        Use the following context to answer: {{context}}
        
        Previous conversation:
        {{history}}
        """
        
        # Update the prompt template with the new system message
        messages = [
            ("system", system_message),
            ("human", "{input}")
        ]
        prompt_template = ChatPromptTemplate.from_messages(messages)
        
        # Create the chain with the updated prompt
        chain = prompt_template | llm | StrOutputParser()
        
        # Get AI response with context and history
        result = chain.invoke({
            "context": context,
            "history": history_text,
            "input": user_input,
            "company_id": company_id
        })
        
        return result
    except Exception as e:
        return f"Error processing request: {str(e)}" 