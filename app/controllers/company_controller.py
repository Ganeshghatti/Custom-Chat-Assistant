import os
import shutil
import time
from app.models.company import Company
from app.utils.vector_store_utils import create_vector_store, vector_stores

def list_companies():
    """List all companies"""
    companies = Company.find_all()
    return {"companies": companies}, 200

def add_company(data):
    """Add a new company"""
    if not data or 'name' not in data or 'file_path' not in data or 'id' not in data:
        return {"error": "Company name, id, and file_path required"}, 400
    
    file_path = data['file_path']
    if not os.path.exists(file_path):
        return {"error": f"File not found at {file_path}"}, 400
    
    # Check if company already exists with the same base ID
    # This is a simplified check - in production you'd want more robust validation
    
    try:
        # Create the company in the database
        company_id, base_id, unique_suffix = Company.create(data)
        
        # Create vector store directory
        vector_store_path = f"vector_stores/{company_id}"
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Create vector store for the new company
        create_vector_store(company_id, data['name'], file_path, vector_store_path)
        
        return {
            "success": True,
            "company_id": company_id,
            "base_id": base_id,
            "unique_suffix": unique_suffix
        }, 201
    except Exception as e:
        # Clean up if there was an error
        Company.delete(company_id)
        return {"error": f"Failed to create vector store: {str(e)}"}, 500

def delete_company(company_id):
    """Delete a company and its associated vector store"""
    try:
        # First, check if the company exists
        company = Company.find_by_id(company_id)
        if not company:
            return {"error": "Company not found"}, 404
        
        # Get the vector store path
        vector_store_path = company.get("vector_store_path", f"vector_stores/{company_id}")
        
        # Remove the vector store from cache if it exists
        if company_id in vector_stores:
            # Get the vector store
            vector_store_data = vector_stores[company_id]
            if "store" in vector_store_data:
                # Close the vector store client to release file handles
                try:
                    vector_store_data["store"]._client.close()
                    print(f"Closed vector store client for {company_id}")
                except Exception as e:
                    print(f"Error closing vector store client: {str(e)}")
            
            # Remove from cache
            del vector_stores[company_id]
            print(f"Removed vector store from cache for {company_id}")
        
        # Delete the company from database
        success = Company.delete(company_id)
        if not success:
            return {"error": "Failed to delete company from database"}, 500
        
        # Delete the vector store directory
        if os.path.exists(vector_store_path):
            # Add a small delay to ensure all file handles are released
            time.sleep(1)
            
            try:
                # Force close any remaining file handles (Windows-specific)
                import gc
                gc.collect()  # Force garbage collection
                
                # Try to delete the directory
                shutil.rmtree(vector_store_path, ignore_errors=True)
                print(f"Deleted vector store at {vector_store_path}")
            except Exception as e:
                print(f"Warning: Could not delete vector store directory: {str(e)}")
                # Continue even if we couldn't delete the directory
                # The company is already removed from the database
        
        return {"success": True, "message": f"Company {company_id} deleted successfully"}, 200
    except Exception as e:
        return {"error": f"Failed to delete company: {str(e)}"}, 500 