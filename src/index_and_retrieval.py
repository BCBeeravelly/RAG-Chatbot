# Import packages
## Utility packages
import scraper  # Custom scraper module
import os
import json
import uuid
from dotenv import load_dotenv
from typing import Dict, List, Tuple

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

# Set the constants
CURRENT_DIR = os.getcwd()
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR  = os.path.join(BASE_DIR, 'data/raw')
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, 'data/sample')
VECTOR_DIR = os.path.join(BASE_DIR, 'data/vectors')
QDRANT_DIR = os.path.join(BASE_DIR, 'data/qdrant')

# Load the environment variables
load_dotenv()

class IndexRetriever:
    def __init__(self,
                 embeddings: VoyageAIEmbeddings,
                 data_dir: str,
                 qdrant_dir: str,
                 child_splitter: RecursiveCharacterTextSplitter,
                 scrape_flag: bool = False,
                 build_vector_flag: bool = False,
                 
                 ):
        """
        Initialize with the embeddings model, data directory, Qdrant directory, 
        and flags for scraping and building the vector DB.
        """
        self.embeddings = embeddings
        self.data_dir = data_dir
        self.qdrant_dir = qdrant_dir
        self.child_splitter = child_splitter
        
        # Setting the flags
        self.scrape_flag = scrape_flag
        self.build_vector_flag = build_vector_flag
        
        
        # These attributes will be set later in the pipeline
        self.documents: List[Document] = []
        self.split_docs: List[Document] = []
        self.vector_store: QdrantVectorStore = None
        
        self.docstore: InMemoryStore = None # For parent document retrieval
        self.parent_retriever: ParentDocumentRetriever = None

    def metadata_extractor(self, record: Dict, metadata: Dict) -> Dict:
        """
        Extracts metadata from the given record.
        """
        
        return {
            'Title': record.get('Title', 'Unknown'),
            'DateSigned': record.get('DateSigned', ''),
            'URL': record.get('URL', '')
        }

    def json_loader(self, json_path: str) -> List[Document]:
        """
        Loads and processes JSON documents from the given path.
        """
        
        loader = JSONLoader(
            file_path=json_path,
            jq_schema='.ExecutiveOrder[]',  # Extract individual EO entries
            content_key='Description',       # Extract the main text content
            metadata_func=self.metadata_extractor
        )
        return loader.load()

    def load_data(self) -> List[Document]:
        """
        Loads data. If scrape_flag is True, scrapes data using scraper;
        otherwise, loads documents from the specified data directory.
        Each loaded document is assigned a unique UUID stored in metadata['_id'].
        """
        if self.scrape_flag:
            print('Scraping data...')
            scraper.scrape(scraper.BASE_URL)
            print('Data scraped successfully.')
        else:
            print('Loading data...')
            
        docs: List[Document] = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_dir, filename)
                doc = self.json_loader(json_path=file_path)
                docs.extend(doc)
                
        # # Assign a unique UUID for each document (store in metadata['_id'])
        # for doc in docs:
        #     doc.metadata["_id"] = str(uuid.uuid4())
        self.documents = docs
        return self.documents

    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], RecursiveCharacterTextSplitter]:
        """
        Splits the given documents into smaller chunks using RecursiveCharacterTextSplitter.
        Returns both the list of child chunks and the splitter (for later use).
        """
        
        print('Splitting documents into child chunks...')
        split_docs = self.child_splitter.split_documents(documents)
        self.split_docs = split_docs
        return self.split_docs, self.child_splitter

    def build_index(self, collection_name: str = "Executive-Orders") -> QdrantVectorStore:
        """
        Builds (or loads) a QdrantVectorStore.
        If build_vector_flag is True, deletes any existing collection and re-creates it,
        then returns the vector store.
        """
        # Create a Qdrant client using the specified local directory for persistence.
        client = QdrantClient(path=self.qdrant_dir)
        
        if self.build_vector_flag:
            print('Building the vector store...')
            # Delete the existing collection if present
            try:
                client.delete_collection(collection_name)
                print('Existing collection deleted.')
            except Exception as e:
                print(f'No existing collection to delete: {e}')
            
            # Set vector parameters (make sure size matches your embedding dimension)
            
            vector_params = VectorParams(size=1024, distance=Distance.COSINE)
            client.create_collection(collection_name=collection_name, vectors_config=vector_params)
        else:
            print('Loading the vector store...')
            
        # Create the QdrantVectorStore. (It will be persisted on disk via qdrant_dir)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings  # This should be compatible with your QdrantVectorStore version
        )
        self.vector_store = vector_store
        return vector_store
    
    # def build_docstore(self, split_docs: List[Document]) -> InMemoryStore:
    #     """
    #     Builds an InMemoryStore for parent document retrieval.
    #     """
    #     print('Building the document store...')
    #     docstore = InMemoryStore()
    #     self.docstore = docstore
    #     return docstore
    
    def build_parent_retriever(self, docstore: InMemoryStore) -> ParentDocumentRetriever:
        """
        Builds a ParentDocumentRetriever using the provided document store.
        """
        print('Building the parent retriever...')
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=InMemoryStore(),
            child_splitter=self.child_splitter
        )
        
        

    def add_documents(self):
        """
        Adds documents (child chunks) to the retriever (i.e. the underlying vector store)
        by using retriever.add_documents(docs, ids) where the ids are taken from each doc's metadata["_id"].
        """
        print('Adding documents to the retriever...')
        self.parent_retriever.add_documents(self.documents, ids=None)
        print('Documents added successfully.')


    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[Document, float]], Document]:
        """
        Performs a similarity search for the given query and retrieves:
         - The top-k child documents (with scores) from the vector store.
         - The parent document as determined by the parent retriever.
        """
        # Use the vector store's similarity search
        results = self.vector_store.similarity_search_with_score(query, k=k)
        # Use the parent retriever's invoke operation to get the parent document.
        parent_document = self.parent_retriever.invoke(query)
        return results, parent_document
    

    def setup_for_retrieval(self):
        """Run setup steps (load data, build index, etc.) without a query."""
        self.load_data()
        self.split_documents(self.documents)
        self.build_index()
        # self.build_docstore(self.split_docs)
        self.build_parent_retriever(self.docstore)
        self.add_documents()

    def full_pipeline(self, query: str, k: int = 5) -> Tuple[List[Tuple[Document, float]], Document]:
        """
        Runs the full pipeline: load data, split documents, build index,
        add documents to the retriever, build docstore and parent retriever, and then retrieve documents.
        Returns the top-k child chunks with their similarity scores and the parent document.
        """
        # 1. Load the parent documents (with UUIDs)
        self.load_data()
        # 2. Split into child chunks
        self.split_documents(self.documents)
        # 3. Build or load the vector store index (persistent via Qdrant)
        self.build_index()
        # # 4. Build the document store for parent retrieval
        # self.build_docstore(self.split_docs)
        # 5. Build the parent retriever
        self.build_parent_retriever(self.docstore)
        # 6. Add the child chunks to the vector store via the retriever mechanism
        self.add_documents()
        # 7. Finally, perform retrieval
        return self.retrieve(query, k=k)
        

# ----------------------------
# Main Execution Example
# ----------------------------
if __name__ == "__main__":
    # Initialize the embeddings model
    voyage_embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        model='voyage-3-large'
    )
    
    # Instantiate the combined class
    index_retriever = IndexRetriever(
        embeddings=voyage_embeddings,
        data_dir=SAMPLE_DATA_DIR,
        qdrant_dir=QDRANT_DIR,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=600, separators=['\n']),
        scrape_flag=False,       # Set True to scrape; False to load from DATA_DIR
        build_vector_flag=True  # Set True to rebuild the vector store; False to load existing
    )
    
    # Run the full pipeline to perform retrieval
    query = "Is there a hiring freeze?"
    results, parent_document = index_retriever.full_pipeline(query, k=5)
    
    # Print the retrieved child documents and their similarity scores
    print("Top 5 child documents retrieved:")
    for i, (doc, score) in enumerate(results):
        title = doc.metadata.get("Title", "Unknown Title")
        print(f"{i+1}. {title}")
        print(f"   Similarity Score: {score}")
    
    # Print the parent document (if retrieved)
    if parent_document:
        print("\nParent Document (first 300 characters):")
        print(parent_document[0].page_content[:300])
    else:
        print("\nNo parent document retrieved.")
