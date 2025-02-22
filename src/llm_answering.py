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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, 'data/sample')
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
            data_dir=SAMPLE_DATA_DIR,
            qdrant_dir=QDRANT_DIR,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size = 600, separators=['\n']),
            scrape_flag=False,
            build_vector_flag=True
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
    
    
        
if __name__ == "__main__":
    agent = LLMAnsweringAgent()
    
    # questions = [
    #     "Is there a hiring freeze?"
    # ]
    
    # for q in questions:
    #     print(f"\nUser: {q}")
    #     response = agent.answer_question(q)
    #     print(f"Assistant: {response}")
    #     print("---" * 20)
    question = "Is there a hiring freeze?"
    child_docs, parent_docs = agent.retrieve_documents(question)
    print(parent_docs)
    print('---' * 20)
    print(child_docs)    