import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app import companies_collection
from app.config import SMTP_EMAIL, SMTP_PASSWORD, ADMIN_EMAIL

def send_email(name, user_email, message, company_id=None):
    """
    Send an email to the company admin
    
    Args:
        name: User's name
        user_email: User's email address
        message: User's message
        company_id: Optional company ID to find the admin's email
    """
    try:
        # Get sender email credentials from environment variables
        sender_email = SMTP_EMAIL
        sender_password = SMTP_PASSWORD
        
        # Get admin email from company data if company_id is provided
        admin_email = None
        if company_id:
            company = companies_collection.find_one({"_id": company_id})
            if company and "admin_email" in company:
                admin_email = company["admin_email"]
        
        # If no admin email found in company data, use default from env
        if not admin_email:
            admin_email = ADMIN_EMAIL
            
        if not sender_email or not sender_password or not admin_email:
            print("Email configuration missing. Check your .env file.")
            return False
            
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = admin_email
        msg['Subject'] = f"New message from {name} via Chatbot"
        
        # Email body
        body = f"""
        Name: {name}
        Email: {user_email}
        
        Message:
        {message}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
            
        print(f"Email sent successfully to admin ({admin_email})")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False 