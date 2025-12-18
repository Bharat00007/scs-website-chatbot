import os
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, vector_store_path="./vector_store"):
        self.vector_store_path = vector_store_path
        os.makedirs(vector_store_path, exist_ok=True)
        
        logger.info(f"Initializing PDFProcessor with vector store at: {vector_store_path}")
        
        try:
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.client = chromadb.PersistentClient(path=vector_store_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Count existing documents
            doc_count = self.collection.count()
            logger.info(f"PDFProcessor initialized. Existing documents: {doc_count}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PDFProcessor: {str(e)}")
            raise
    
    def get_collection(self):
        """Get the ChromaDB collection"""
        return self.collection
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF and split into chunks"""
        text_chunks = []
        
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"PDF has {total_pages} pages")
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Split text into chunks
                        chunks = self._split_text_into_chunks(text, chunk_size=500)
                        text_chunks.extend(chunks)
                    
                    # Log progress every 10 pages
                    if page_num % 10 == 0 and page_num > 0:
                        logger.info(f"Processed {page_num}/{total_pages} pages")
                
                logger.info(f"Extracted {len(text_chunks)} text chunks from {total_pages} pages")
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        
        return text_chunks
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def process_pdf(self, pdf_path: str):
        """Process a PDF file and store embeddings"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text chunks
            text_chunks = self.extract_text_from_pdf(pdf_path)
            
            if not text_chunks:
                logger.warning("No text extracted from PDF")
                return
            
            logger.info(f"Generating embeddings for {len(text_chunks)} chunks...")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(text_chunks).tolist()
            
            # Create IDs for each chunk
            ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
            
            # Add metadata (page numbers, etc.)
            metadatas = [{"source": os.path.basename(pdf_path)} for _ in range(len(text_chunks))]
            
            # Add to collection
            self.collection.add(
                documents=text_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(text_chunks)} chunks in vector database")
            logger.info(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def search_context(self, query: str, top_k: int = 3) -> str:
        """Search for relevant context based on query"""
        try:
            if self.collection.count() == 0:
                logger.info("No documents in collection")
                return ""
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # Combine results
            if results['documents']:
                context_chunks = []
                for i, doc in enumerate(results['documents'][0]):
                    source = results['metadatas'][0][i].get('source', 'Unknown') if results['metadatas'] else 'Unknown'
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    # Only include relevant chunks (distance < 1.0 for cosine similarity)
                    if distance < 1.0:  # Cosine similarity threshold
                        context_chunks.append(f"[From: {source}]\n{doc}")
                
                context = "\n\n---\n\n".join(context_chunks)
                logger.info(f"Found {len(context_chunks)} relevant chunks for query")
                return context
            
            return ""
            
        except Exception as e:
            logger.error(f"Error searching context: {str(e)}")
            return ""