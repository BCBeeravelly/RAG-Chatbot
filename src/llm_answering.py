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
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough



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
class AnsweringAgent:
    """
    A class to interact with the LLM for answering questions about executive orders.
    """
    def __init__(self, llm_model = 'gpt-4o-mini', temperature = 0):
        
        self.llm_model = llm_model
        self.temperature = temperature
        self.user_query = None
        
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
            build_vector_flag=False
        )
        
        self.retriever.setup_for_retrieval()
        
        self.llm = ChatOpenAI(
            model = self.llm_model,
            temperature = self.temperature,
            streaming = True
        )
        self.child_docs = None
        self.parent_docs = None
        
        # Add conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )
              
    
    def retrieve_documents(self, user_query):
        """
        Retrieve relevant documents based on the user's query.
        """
        # Retrieve relevant documents
        self.user_query = user_query
        print(f"Retrieving documents for query: {self.user_query}")
        self.child_docs, self.parent_docs = self.retriever.retrieve(self.user_query)
        return self.child_docs, self.parent_docs
    
    def answering_agent(self):
        '''Answers with conversation context'''
        context = "\n\n".join([doc.page_content for doc in self.parent_docs]) if self.parent_docs else "No relevant documents found"
        
        # Create conversation-aware prompt
        template = """You're an assistant for executive order questions. Use:
        - Chat history: {chat_history}
        - Context: {context}
        
        Current Question: {question}
        
        If needed, ask follow-up questions. Be helpful and precise. Say "I don't know" if unsure."""
        
        prompt = PromptTemplate.from_template(template)
        
        # Build conversation chain
        conversation_chain = (
            RunnablePassthrough.assign(
                context=lambda _: context,
                chat_history=lambda x: self.memory.load_memory_variables(x)["chat_history"]
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return conversation_chain.invoke({"question": self.user_query})
    
    def chat_loop(self):
        """Run continuous chat interface"""
        print("Executive Order Assistant. Type 'exit' to end.")
        while True:
            try:
                question = input("\nYou: ")
                if question.lower() in ['exit', 'quit']:
                    break
                
                # Retrieve docs and generate answer
                self.retrieve_documents(question)
                answer = self.answering_agent()
                
                # Update memory
                self.memory.save_context(
                    {"question": question},
                    {"answer": answer}
                )
                
                print(f"\nAssistant: {answer}")
                
            except KeyboardInterrupt:
                print("\nSession ended.")
                break
    
    
        
# Update the main block
if __name__ == "__main__":
    agent = AnsweringAgent()
    agent.chat_loop()
    