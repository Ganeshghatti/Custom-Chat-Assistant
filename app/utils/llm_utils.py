from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.utils.vector_store_utils import get_vector_store, embeddings
from app.config import LLM_MODEL, SIMILARITY_THRESHOLD

# Initialize the LLM with streaming enabled and specific parameters
llm = ChatGroq(
    model=LLM_MODEL,
    streaming=True,
    temperature=0.7,
)

def chat(user_input, company_id, conversation_history=None):
    """Process a chat message and return a response generator for streaming"""
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
        
        print(f"context {context}")
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
        5. CRITICAL: Your responses must be complete and meaningful. Always finish your thoughts and sentences.
        6. Keep responses brief (under 200 tokens) but ensure they are complete and coherent.
        7. Never end mid-sentence or with an incomplete thought.
        
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
        
        # Return the streaming generator
        return chain.stream({
            "context": context,
            "history": history_text,
            "input": user_input,
            "company_id": company_id
        })
    except Exception as e:
        # For errors, yield the error message as a single item
        def error_generator():
            yield f"Error processing request: {str(e)}"
        return error_generator() 