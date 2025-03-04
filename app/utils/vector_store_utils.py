import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app import companies_collection
from app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR

# Initialize Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Vector stores cache
vector_stores = {}

def initialize_vector_stores():
    """Initialize vector stores for all companies in the database"""
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
                vector_store_path = f"{VECTOR_STORE_DIR}/{company_id}"
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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