import os
import torch
import faiss
import numpy as np
import streamlit as st
from typing import List

# Hugging Face Transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class MedicalChatbot:
    def __init__(self, 
                 pdf_path: str, 
                 model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        # Force CPU usage
        self.device = "cpu"
        torch.set_default_device(self.device)
        
        # Knowledge Base Setup
        self.pdf_path = pdf_path
        self.documents = []
        self.embeddings = None
        self.index = None
        
        # Embedding Model (CPU-friendly)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Load and process PDF
        self.load_medical_knowledge()
        
        # LLM Setup
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",  # Explicitly use CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=300
            )
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise
    
    def load_medical_knowledge(self):
        try:
            # Load PDF
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            self.documents = text_splitter.split_documents(documents)
            
            # Generate embeddings
            texts = [doc.page_content for doc in self.documents]
            self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Create FAISS index
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
        
        except Exception as e:
            st.error(f"Error processing medical knowledge: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        try:
            # Embed query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            # Search in index
            D, I = self.index.search(np.array([query_embedding]), top_k)
            
            # Retrieve top results
            context = "\n".join([self.documents[i].page_content for i in I[0]])
            return context
        
        except Exception as e:
            st.error(f"Context retrieval error: {e}")
            return "No relevant context found."
    
    def generate_response(self, query: str) -> str:
        # Retrieve contextual information
        context = self.retrieve_context(query)
        
        # Construct prompt with context
        prompt = f"""
        You are a medical information assistant. 
        Context from medical encyclopedia: {context}
        
        User Question: {query}
        
        Provide a clear, informative, and concise medical explanation. 
        Always include a disclaimer that this is for informational purposes only.
        
        Detailed Response:
        """
        
        # Generate response
        try:
            response = self.generator(
                prompt, 
                do_sample=True, 
                top_k=10, 
                num_return_sequences=1
            )[0]['generated_text']
            
            return response.split("Detailed Response:")[-1].strip()
        
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return "I'm unable to generate a response at the moment."

def main():
    st.title('Medical Information Chatbot (CPU Edition)')
    
    # PDF path (update with your actual path)
    PDF_PATH = 'gale.pdf'
    
    # Initialize chatbot (cached)
    @st.cache_resource
    def load_chatbot():
        return MedicalChatbot(PDF_PATH)
    
    try:
        chatbot = load_chatbot()
    except Exception as e:
        st.error(f"Failed to load chatbot: {e}")
        return
    
    # Chat interface
    user_query = st.text_input('Ask a medical question:')
    
    if user_query:
        # Generate and display response
        with st.spinner('Generating response...'):
            try:
                response = chatbot.generate_response(user_query)
                st.markdown(f"**Response:** {response}")
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Sidebar information
    st.sidebar.header('About')
    st.sidebar.info(
        'This medical chatbot provides information based on the Gale Encyclopedia of Medicine. '
        'Always consult a healthcare professional for personalized medical advice.'
    )

if __name__ == '__main__':
    main()
