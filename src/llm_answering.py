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
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
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
        '''
        Answers the user's query using the retrieved documents.
        '''

        # Combine parent documents' content
        context = "\n\n".join([doc.page_content for doc in self.parent_docs])

        # ReAct prompt template
        template = """Answer the question using ONLY this context. Think step-by-step.
        If unsure, say you don't know. Use this format:

        Question: {question}

        Context: {context}

        Thought: First, I need to...
        Action: check_context
        Action Input: [relevant part from question]
        Observation: [context information]
        ... (repeat Thought/Action/Observation as needed)
        Final Answer: [concise answer]"""

        prompt = PromptTemplate.from_template(template)
        
        # Set up agent pipeline (without actual tools)
        agent = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                question=lambda x: x["question"]
            )
            | prompt
            | self.llm
            | ReActSingleInputOutputParser()
        )

        # Execute agent
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=[],  # Will add tools later
            verbose=True,
            handle_parsing_errors=True
        )

        result = agent_executor.invoke({"question": self.user_query})
        return result["output"]
    
    
        
if __name__ == "__main__":
    agent = AnsweringAgent()
    question = input("Enter your question: ")
    agent.retrieve_documents(question)
    answer = agent.answering_agent()
    print(f"Answer: {answer}")
    