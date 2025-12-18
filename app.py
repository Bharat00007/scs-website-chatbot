import os
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import traceback
from pdf_processor import PDFProcessor

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS - allow your GoDaddy domain
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://semicom-consultancy.com",  # Your GoDaddy domain
            "https://semicom-consultancy.com", # HTTPS version
            "http://localhost:5500",           # For local testing
            "http://127.0.0.1:5500",           # For local testing
            "*"                                # Allow all for testing
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize PDF processor
print("Initializing PDF processor...")
try:
    pdf_processor = PDFProcessor()
    print("PDF processor initialized successfully")
    
    # AUTO-LOAD PDFs from data folder on startup
    print("\n=== AUTO-LOADING PDFs FROM DATA FOLDER ===")
    data_folder = "data"
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files in {data_folder}/")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_folder, pdf_file)
            try:
                print(f"Processing: {pdf_file}")
                pdf_processor.process_pdf(pdf_path)
                print(f"✓ Successfully loaded: {pdf_file}")
            except Exception as e:
                print(f"✗ Failed to load {pdf_file}: {str(e)}")
    else:
        print(f"Data folder '{data_folder}' not found. Creating it...")
        os.makedirs(data_folder, exist_ok=True)
        
    # Get document count
    if pdf_processor.collection:
        doc_count = pdf_processor.collection.count()
        print(f"\n✅ Total documents in database: {doc_count}")
    
except Exception as e:
    print(f"Error initializing PDF processor: {str(e)}")
    pdf_processor = None

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get chatbot status and document info"""
    try:
        if not pdf_processor:
            return jsonify({
                "status": "pdf_processor_not_ready",
                "message": "PDF processor is not initialized"
            }), 500
        
        collection = pdf_processor.get_collection()
        doc_count = collection.count() if collection else 0
        
        return jsonify({
            "status": "ready",
            "documents_loaded": doc_count,
            "nvidia_api": "configured" if NVIDIA_API_KEY else "not_configured",
            "backend": "running"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """Home page for Render health check"""
    return jsonify({
        "status": "online",
        "service": "RAG Chatbot API",
        "endpoints": {
            "health": "/api/health (GET)",
            "upload": "/api/upload-pdf (POST)",
            "chat": "/api/chat (POST)"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "pdf_processor": "ready" if pdf_processor else "not_ready",
        "nvidia_api": "configured" if NVIDIA_API_KEY else "not_configured"
    }), 200

@app.route('/api/upload-pdf', methods=['POST', 'OPTIONS'])
def upload_pdf():
    """Endpoint to upload and process PDF files"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must be a PDF"}), 400
    
    # Check file size (limit to 10MB)
    pdf_file.seek(0, 2)  # Seek to end
    file_size = pdf_file.tell()
    pdf_file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 400
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_file.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    try:
        # Process the PDF
        if not pdf_processor:
            return jsonify({"error": "PDF processor not ready"}), 500
        
        pdf_processor.process_pdf(tmp_path)
        
        # Get document count
        collection = pdf_processor.get_collection()
        doc_count = collection.count() if collection else 0
        
        return jsonify({
            "success": True,
            "message": "PDF processed successfully",
            "filename": pdf_file.filename,
            "documents_processed": doc_count
        }), 200
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint with RAG"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received chat request: {user_message[:50]}...")
        
        # 1. Retrieve relevant context from PDFs
        context = ""
        if pdf_processor:
            context = pdf_processor.search_context(user_message, top_k=3)
        
        # 2. Prepare the prompt with context
                # 2. Prepare the prompt with context
        if context and context.strip():
            prompt = f"""You are a customer service representative for SEMISHARE (SemiShare). You are talking to a customer. Use the following information from our company documents to answer the customer's question.

Company Information:
{context}

Customer Question: {user_message}

IMPORTANT INSTRUCTIONS:
1. Respond in FIRST PERSON as if YOU ARE THE COMPANY ("We provide...", "Our products include...", "I can help you with...")
2. Do NOT say "Based on the context" or "According to the documents" - just answer naturally
3. Be helpful, friendly, and professional
4. If you don't have the information, say: "I'd be happy to help with that! Let me check with our team for more details."
5. Never mention that you're an AI or that you're reading from documents
6. Speak naturally like a human customer service agent

Now, answer the customer's question:"""
        else:
            prompt = f"""You are a customer service representative for SEMISHARE (SemiShare). A customer is asking you a question.

Customer Question: {user_message}

IMPORTANT INSTRUCTIONS:
1. Respond in FIRST PERSON as if YOU ARE THE COMPANY ("We provide...", "Our products include...", "I can help you with...")
2. Be helpful, friendly, and professional
3. If you need more details to answer, ask clarifying questions
4. Never mention that you're an AI

Now, answer the customer's question:"""        
        # 3. Call NVIDIA API
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
                "model": "mistralai/mistral-large-3-675b-instruct-2512",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a customer service representative for SEMISHARE (SemiShare), a semiconductor testing equipment company. You help customers with product information, specifications, and inquiries. Always speak in first person as the company ('we', 'our', 'us'). Be professional, helpful, and friendly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        print("Calling NVIDIA API...")
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            assistant_reply = result['choices'][0]['message']['content']
            
            return jsonify({
                "success": True,
                "response": assistant_reply,
                "has_context": bool(context and context.strip())
            })
        else:
            print(f"NVIDIA API error: {response.status_code} - {response.text}")
            return jsonify({
                "error": "Failed to get response from AI service",
                "details": response.text[:200] if response.text else "No response"
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to AI service. Please check your internet connection."}), 503
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about processed documents"""
    try:
        if not pdf_processor:
            return jsonify({"error": "PDF processor not ready"}), 500
        
        collection = pdf_processor.get_collection()
        if not collection:
            return jsonify({"documents": [], "count": 0})
        
        # Get count and some metadata
        count = collection.count()
        
        # Get a few sample documents (first 5)
        results = collection.get(limit=min(5, count))
        
        return jsonify({
            "count": count,
            "documents": results.get('documents', [])[:3] if results else []
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)