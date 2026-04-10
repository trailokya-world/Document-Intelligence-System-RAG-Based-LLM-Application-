from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
import ollama
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Database Directory
persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# Loading database
db = Chroma(
    persist_directory=persist_directory,
    embedding_function= embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# search query
query = "Which company did NVIDIA acquire to enter the mobile processor market?"
 
retriever = db.as_retriever()
# retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_doc = retriever.invoke(query)

print("*"*40)
print(f"User Query: {query}")
print("*"*40)
for i, doc in enumerate(relevant_doc):
    print(f"---{i+1}--- \n {doc.page_content} \n")
    
    
# Combine the query and relevent document contents:
combined_input = f""" based on following document please answer this question: {query} 
documents:
{chr(10).join([f"- {doc.page_content}"  for doc in relevant_doc])} 
please provide clear and helpful answer using only given documents, if can't find the answer say i don't have enough information to answer this question based on the provided documents 
"""

messages = [
    
    SystemMessage(content="You are the helpfull assistant"),
    HumanMessage(content=combined_input)
    
    ]
    

result = ollama.chat(
    model="gemma3:4b",
    messages=[
            {
                "role": "system",
                "content": "You are the helpfull assistant."
            },
            {
                "role": "user",
                "content": combined_input
            }
        ]
    )

# result = model1.invoke(messages)
final_result = result["message"]["content"]
print("\nContent only:\n")
print(final_result)