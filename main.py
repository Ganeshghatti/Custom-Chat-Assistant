from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pymongo import MongoClient
import time

app = Flask(__name__)
load_dotenv()

# Initialize MongoDB connection
mongo_uri = "mongodb+srv://ganeshghatti6:5IPx4NHMYuW8pg72@chatbot.ke4gb.mongodb.net/?retryWrites=true&w=majority&appName=chatbot"
mongo_client = MongoClient(mongo_uri)
db = mongo_client["chatbot"]
companies_collection = db["companies"]

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192")

# Initialize Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Vector stores cache
vector_stores = {}

# Initialize vector stores for companies that don't have them
def initialize_vector_stores():
    print("Initializing vector stores...")
    companies = companies_collection.find({})
    for company in companies:
        company_id = company["_id"]
        company_name = company["name"]
        file_path = company.get("file_path")
        vector_store_path = company.get("vector_store_path")
        
        # Skip if file doesn't exist
        if not file_path or not os.path.exists(file_path):
            print(f"Warning: File not found for company {company_name}")
            continue
        
        # Create vector store if it doesn't exist
        if not vector_store_path or not os.path.exists(vector_store_path):
            try:
                # Create a directory for vector store
                vector_store_path = f"vector_stores/{company_id}"
                os.makedirs(vector_store_path, exist_ok=True)
                print(f"Created directory for vector store: {vector_store_path}")
                
                # Create vector store
                create_vector_store(company_id, company_name, file_path, vector_store_path)
                
                # Update the company record with vector store path
                companies_collection.update_one(
                    {"_id": company_id},
                    {"$set": {"vector_store_path": vector_store_path}}
                )
                
                print(f"Created vector store for {company_name}")
            except Exception as e:
                print(f"Error creating vector store for {company_name}: {str(e)}")
        else:
            print(f"Vector store already exists for {company_name}")

def create_vector_store(company_id, company_name, file_path, vector_store_path):
    """Create a vector store for a company based on its text file"""
    print(f"Creating vector store for {company_name}...")
    # Load company information from file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            company_info = file.read()
        print(f"Successfully loaded company information for {company_name}")
    except Exception as e:
        print(f"Error loading company information for {company_name}: {e}")
        company_info = ""
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_text(company_info)
    print(f"Split company information into {len(chunks)} chunks.")
    
    # Create the vector store from texts
    vector_store = Chroma.from_texts(
        texts=chunks,
        collection_name=f"company_{company_id}",
        persist_directory=vector_store_path,
        embedding=embeddings
    )
    # Save the vector store
    vector_store.persist()
    print(f"Vector store for {company_name} has been persisted.")
    
    # Cache the vector store
    vector_stores[company_id] = {
        "store": vector_store,
        "name": company_name
    }

def get_vector_store(company_id):
    """Get vector store for a specific company"""
    print(f"Retrieving vector store for company ID: {company_id}")
    if company_id in vector_stores:
        print(f"Vector store found in cache for company ID: {company_id}")
        return vector_stores[company_id]
    
    # If not in cache, try to load it
    company = companies_collection.find_one({"_id": company_id})
    if not company:
        raise ValueError(f"Company with ID {company_id} not found")
    
    company_name = company.get("name", "Unknown Company")
    vector_store_path = company.get("vector_store_path")
    
    if not vector_store_path or not os.path.exists(vector_store_path):
        raise ValueError(f"Vector store not found for company {company_name}")
    
    # Load the vector store
    vector_store = Chroma(
        collection_name=f"company_{company_id}",
        persist_directory=vector_store_path,
        embedding_function=embeddings
    )
    
    # Cache the vector store
    vector_stores[company_id] = {
        "store": vector_store,
        "name": company_name
    }
    
    print(f"Vector store for {company_name} loaded successfully.")
    return vector_stores[company_id]

def chat(user_input, company_id, conversation_history=None):
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
        similarity_threshold = 1.25  # Adjust this threshold as needed (lower = more strict)
        print(f"docs {docs}")
        print(f"similarity_threshold {similarity_threshold}")
        
        relevant_docs = [doc for doc, score in docs if score <= similarity_threshold]
    
        if not relevant_docs:
            return f"I'm sorry, but I don't have enough information to answer that question. I'm a {company_name} assistant and can only provide information about {company_name}'s products, services, and company details. Could you ask something related to {company_name}?"
        
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
            "input": user_input
        })
        
        return result
    except Exception as e:
        return f"Error processing request: {str(e)}"

def generate_unique_suffix():
    """Generate a unique suffix using timestamp and random number"""
    return str(int(time.time() * 1000))[-6:]  # Last 6 digits of current timestamp in milliseconds

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    print(f"Chat endpoint called with data: {data}")
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    if 'company_id' not in data:
        return jsonify({"error": "No company_id provided"}), 400
    
    user_input = data['message']
    company_id = data['company_id']
    conversation_history = data.get('conversation_history', [])
    
    # Limit to last 10 exchanges to maintain context without overwhelming
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    response = chat(user_input, company_id, conversation_history)
    
    # Return the response
    return jsonify({
        "response": response
    })

@app.route('/companies', methods=['GET'])
def list_companies():
    """Endpoint to list all available companies"""
    companies = list(companies_collection.find({}, {"_id": 1, "name": 1}))
    print(f"Listing companies: {companies}")
    return jsonify({"companies": companies})

@app.route('/company', methods=['POST'])
def add_company():
    """Endpoint to add a new company"""
    data = request.json
    if not data or 'name' not in data or 'file_path' not in data or 'id' not in data:
        return jsonify({"error": "Company name, id, and file_path required"}), 400
    
    file_path = data['file_path']
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found at {file_path}"}), 400
    
    # Get base ID from user and append unique suffix
    base_id = data['id']
    unique_suffix = generate_unique_suffix()
    company_id = f"{base_id}_{unique_suffix}"
    
    # Check if company already exists
    existing = companies_collection.find_one({"_id": company_id})
    if existing:
        return jsonify({"error": "Company with this ID already exists"}), 409
    
    # Create vector store directory
    vector_store_path = f"vector_stores/{company_id}"
    os.makedirs(vector_store_path, exist_ok=True)
    
    # Insert new company
    company_data = {
        "_id": company_id,
        "name": data['name'],
        "file_path": file_path,
        "vector_store_path": vector_store_path
    }
    companies_collection.insert_one(company_data)
    
    # Create vector store for the new company
    try:
        create_vector_store(company_id, data['name'], file_path, vector_store_path)
        return jsonify({
            "success": True,
            "company_id": company_id,
            "base_id": base_id,
            "unique_suffix": unique_suffix
        })
    except Exception as e:
        companies_collection.delete_one({"_id": company_id})
        return jsonify({"error": f"Failed to create vector store: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return "Multi-Company Support Bot API is running. Use the /chat endpoint to interact."

if __name__ == '__main__':
    # Make sure vector_stores directory exists
    os.makedirs("vector_stores", exist_ok=True)
    
    # Initialize vector stores for companies that don't have them
    initialize_vector_stores()
    
    app.run(debug=True, host='0.0.0.0', port=5000)