from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import ollama

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceBgeEmbeddings(model_name ="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

chat_history = []

def ask_question(user_question):
    
 
    
    if chat_history:
        
        result = ollama.chat(model = "Llama3.1",
            messages=[
                
                {
                    "role" : "system",
                    "content":"you are the usefull assistant"
                },
                {
                "role": "user",
                "content": "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question." + "\n".join(chat_history)
            }    
            ])
        search_question = result["message"]["content"]
        print("*"*20)
        print(f"\n new_question: {search_question}")      
    else:
        
        search_question = user_question
        
    retriver = db.as_retriever(search_kwargs={"k": 3}) 
    docs = retriver.invoke(search_question)

    print(f"found {len(docs)} relevent document")

    for i, doc in enumerate(docs, 1):
        line = doc.page_content.split("\n")[:2]
        preview = "\n".join(line)
        print(f"preview: {i}: {preview}...")
        
    # final prompt
    combined_input = f""" based on following document please answer this question: {search_question} 
    documents:
    {chr(10).join([f"- {doc.page_content}"  for doc in docs ])} 
    please provide clear and helpful answer using only given documents, if can't find the answer say i don't have enough information to answer this question based on the provided documents 
    """
        


    result = ollama.chat(model = "Llama3.1",
                messages=[
                    
                    {
                        "role" : "system",
                        "content":"you are the usefull assistant"
                    },
                    {
                    "role": "user",
                    "content": combined_input
                }    
                ])
    
    answer = result["message"]["content"]
    print("*"*20)
    print(f"Model answer: {answer}") 
    
    chat_history.append(user_question)
    chat_history.append(answer)
     
     
def start_chat():
    
    
    print('\nAsk me quetion and type quit to exit')

    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)


if __name__ == "__main__":
    start_chat()
        
   
     
        
        
        
        
        