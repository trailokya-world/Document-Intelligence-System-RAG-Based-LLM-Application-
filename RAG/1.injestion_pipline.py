import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def load_documents(docs_path="docs"):
    
    """load all the text file from docs"""
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The Directory {docs_path} is not exists")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
        "encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f" No .text files found in {docs_path}")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocuments {i+1}:")
        print(f" Source: {doc.metadata["source"]}")
        print(f" Content length: {len(doc.page_content)}")
        print(f" Content preview: {doc.page_content[:100]}")
        print(f" Metadata: {doc.metadata}")
            
    return documents

def split_documents(documents, chunk_size = 1000, chunk_overlap = 0):
    
    text_splitter = CharacterTextSplitter(
        chunk_size= chunk_size, 
        chunk_overlap= chunk_overlap
        )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        
        for i, chunk in enumerate(chunks[:5]):
            
            print(f"--- chunk{i+1} ---")
            print(f"Source: {chunk.metadata["source"]}")
            print(f"Length: {len(chunk.page_content)}")
            print("content")
            print(chunk.page_content)   
            
        if len(chunks)>5:
            print(f"--- {len(chunks)-5} more chunks---") 
            
    return chunks       
    
def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    
    """ create and persist chromaBD vector store """
    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    print("___finished creating vectore store___")
    print(f"___ vector store is created and saved to 1 {persist_directory}")
    
    return vector_store    
    
def main():
    #1.Loading Files 
    documents = load_documents()
    # print(len(documents))
    #2.Chunkking the files
    chunks = split_documents(documents)
    #3.Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)
    
        
if __name__ == "__main__":
    main()
    
# charging:75%, screentime:1h28m, time:2:49 
# 50 