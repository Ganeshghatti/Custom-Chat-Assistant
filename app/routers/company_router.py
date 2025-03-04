from flask import Blueprint, request, jsonify
from app.controllers.company_controller import list_companies, add_company, delete_company

company_bp = Blueprint('company', __name__)

@company_bp.route('/companies', methods=['GET'])
def companies_endpoint():
    """Endpoint to list all available companies"""
    response, status_code = list_companies()
    return jsonify(response), status_code

@company_bp.route('/company', methods=['POST'])
def add_company_endpoint():
    """Endpoint to add a new company"""
    data = request.json
    response, status_code = add_company(data)
    return jsonify(response), status_code

@company_bp.route('/company/<string:company_id>', methods=['DELETE'])
def delete_company_endpoint(company_id):
    """Endpoint to delete a company"""
    response, status_code = delete_company(company_id)
    return jsonify(response), status_code 