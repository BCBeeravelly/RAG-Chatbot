# LangChain packages
from langchain.storage import InMemoryStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


## Utility packages
from index_and_retrieval import IndexRetriever
import os
import json
import uuid
from dotenv import load_dotenv
from typing import Dict, List, Tuple


# Set the directory path

CURRENT_DIR = os.getcwd()
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR  = os.path.join(BASE_DIR, 'data/raw')
VECTOR_DIR = os.path.join(BASE_DIR, 'data/vectors')
QDRANT_DIR = os.path.join(BASE_DIR, 'data/qdrant')

# Load environment variables
load_dotenv()
class LLMAnsweringAgent:
    """
    A class to interact with the LLM for answering questions about executive orders.
    """
    def __init__(self, llm_model = 'gpt-4o-mini', temperature = 0):
        
        print('Initializing LLMAnsweringAgent...')
        self.voyage_embeddings = VoyageAIEmbeddings(
            voyage_api_key = os.getenv('VOYAGE_API_KEY'),
            model = 'voyage-3-large'
        )
        
        # Initialize the retriever
        self.retriever = IndexRetriever(
            embeddings=self.voyage_embeddings,
            data_dir=DATA_DIR,
            qdrant_dir=QDRANT_DIR,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size = 600, separators=['\n']),
            scrape_flag=False,
            build_vector_flag=False
        )
        
        self.retriever.setup_for_retrieval()
        
        self.llm = ChatOpenAI(
            model = llm_model,
            temperature = temperature
        )
    
    def retrieve_documents(self, user_query):
        """
        Retrieve relevant documents based on the user's query.
        """
        # Retrieve relevant documents
        print(f"Retrieving documents for query: {user_query}")
        child_docs, parent_docs = self.retriever.retrieve(user_query)
        
        return child_docs, parent_docs
    
    def generate_answer(self, user_query, child_docs, parent_docs):
        """
        Generate an answer to the user's question using the LLM.
        """
        # Create a prompt for the LLM
        prompt = f"""
        You are an expert in US executive orders. Your task is to answer questions based on the information provided in the executive orders. 
        If the information is not available in the provided documents, respond with "I don't know".

        Question: {user_query}

        Context:
        """
        
        # Add context from child documents
        for doc in child_docs:
            prompt += f"\n\n{doc.page_content}\n\n"
        
        # Add context from parent documents
        for doc in parent_docs:
            prompt += f"\n\n{doc.page_content}\n\n"
        
        # Generate an answer using the LLM
        response = self.llm([SystemMessage(content=prompt)])
        
        return response.content.strip()
        
if __name__ == "__main__":
    agent = LLMAnsweringAgent()
    user_query = "Is there a hiring freeze?"
    child_docs, parent_docs = agent.retrieve_documents(user_query)
    print(f"Number of child documents: {len(child_docs)}")
    print(f"Number of parent documents: {len(parent_docs)}")
    print(f"Child docs: {child_docs}")
    print(f"Parent docs: {parent_docs[0].page_content}")