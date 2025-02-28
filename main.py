from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import os
import numpy as np

app = Flask(__name__)
load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192")

# Initialize Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Social Hardware company information
company_info = """
Founded in 2017, Social Hardware is a Bengaluru-based startup dedicated to enhancing and protecting lives through innovative technology. Initially focused on assistive devices for underserved communities, the company gained national attention in 2021 with its collaboration with Cadbury Celebrations for the #MyFirstRakhi campaign, providing prostheses that enabled children with upper limb differences to fully engage in the Raksha Bandhan tradition. The prostheses, which offered a sense of touch, were provided at no cost, highlighting Social Hardware's commitment to inclusivity. The campaign garnered widespread attention, reaching 2.7 crore views on YouTube, and was praised for promoting inclusivity in cultural practices, marking a significant step in making assistive technology more accessible and integrated into cultural traditions in India. Building on its expertise in bionics and prosthetics, the company expanded into field robotics, developing the Eclipse Remote Systems—teleoperated platforms designed for high-risk tasks like bomb disposal and search and rescue. Featuring immersive augmented reality (AR) controls, these systems ensure seamless human-machine interaction, allowing operators to perform hazardous tasks with precision and safety from a distance. Social Hardware is actively seeking industry partners to co-develop new models in the Eclipse Series—each crafted to address specialized needs in high-risk sectors. Organizations interested in exploring how Social Hardware's Eclipse Series can transform their approach to safety and efficiency are invited to schedule a demo or discuss potential collaborations.
"""

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunks = text_splitter.split_text(company_info)

# Create FAISS index
chroma_db_path = "social_hardware_info"

# Check if the vector store already exists
if os.path.exists(chroma_db_path) and os.path.isdir(chroma_db_path) and len(os.listdir(chroma_db_path)) > 0:
    # Load existing vector store
    vector_store = Chroma(collection_name="social_hardware_info",
                          persist_directory=chroma_db_path,
                          embedding_function=embeddings)
    print("Loaded existing Chroma DB")
else:
    # Create the vector store from texts
    vector_store = Chroma.from_texts(
        texts=chunks,
        collection_name="social_hardware_info",
        persist_directory=chroma_db_path,
        embedding=embeddings
    )
    # Save the vector store
    vector_store.persist()
    print("Created new Chroma DB")

def chat(user_input, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    # Create embedding for the user query
    query_embedding = embeddings.embed_query(user_input)
    
    # Retrieve relevant context based on the user query
    docs = vector_store.similarity_search_by_vector(query_embedding, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Format conversation history for context
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = "Assistant" if msg['role'] == 'assistant' else "User"
            history_text += f"{role}: {msg['content']}\n"
    
    # Create a more focused system message
    system_message = """You are a specialized support assistant for Social Hardware, a Bengaluru-based company focused on assistive devices and field robotics.
    
    IMPORTANT GUIDELINES:
    1. ONLY answer questions related to Social Hardware, its products, services, or company information
    2. If asked about unrelated topics, politely redirect the conversation to Social Hardware
    3. Use the provided context and conversation history to give accurate, helpful responses
    4. Be concise and professional in your responses
    
    Use the following context to answer: {context}
    
    Previous conversation:
    {history}
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

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_input = data['message']
    conversation_history = data.get('conversation_history', [])
    
    # Limit to last 10 exchanges to maintain context without overwhelming
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    response = chat(user_input, conversation_history)
    
    # Return the response
    return jsonify({
        "response": response
    })

@app.route('/', methods=['GET'])
def home():
    return "Social Hardware Support Bot API is running. Use the /chat endpoint to interact."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)