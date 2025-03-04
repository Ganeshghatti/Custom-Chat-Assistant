from app import companies_collection
import time

class Company:
    @staticmethod
    def find_all(projection=None):
        """Get all companies from the database"""
        if projection is None:
            projection = {"_id": 1, "name": 1, "file_path": 1}
        return list(companies_collection.find({}, projection))
    
    @staticmethod
    def find_by_id(company_id):
        """Find a company by ID"""
        return companies_collection.find_one({"_id": company_id})
    
    @staticmethod
    def create(data):
        """Create a new company"""
        # Generate a unique ID
        base_id = data['id']
        unique_suffix = str(int(time.time() * 1000))[-6:]  # Last 6 digits of timestamp
        company_id = f"{base_id}_{unique_suffix}"
        
        # Create company document
        company_data = {
            "_id": company_id,
            "name": data['name'],
            "file_path": data['file_path'],
            "vector_store_path": f"vector_stores/{company_id}",
            "admin_email": data.get('admin_email')
        }
        
        # Insert into database
        companies_collection.insert_one(company_data)
        
        return company_id, base_id, unique_suffix
    
    @staticmethod
    def update(company_id, update_data):
        """Update a company"""
        companies_collection.update_one(
            {"_id": company_id},
            {"$set": update_data}
        )
    
    @staticmethod
    def delete(company_id):
        """Delete a company"""
        # Delete from database
        result = companies_collection.delete_one({"_id": company_id})
        return result.deleted_count > 0 