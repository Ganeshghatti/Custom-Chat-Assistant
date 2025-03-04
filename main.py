from app import app
from app.utils.vector_store_utils import initialize_vector_stores

if __name__ == '__main__':
    # Initialize vector stores for companies that don't have them
    initialize_vector_stores()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)