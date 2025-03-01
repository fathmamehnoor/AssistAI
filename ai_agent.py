import os
import sys
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_collection(name="support_knowledge")

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key (ensure it is set correctly)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY is not set in the environment variables.")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)
model_name = "gpt-4o"

# Store previous interaction for session memory
context = []

def retrieve_knowledge(query):
    
    results = collection.query(query_texts=[query], n_results=1)
    return results["documents"][0] if results["documents"] else "No relevant information found."


def chat(user_query):

    global context
    
    knowledge = retrieve_knowledge(user_query)

    conversation_history = "\n".join(context[-3:])

    prompt = f"""You are a helpful customer support AI. Answer the customer's query using this knowledge:\n\n
    {knowledge}\n
    Here is the recent conversation history to maintain context:
    {conversation_history}\n
    Customer: {user_query}\n
    AI:"""

    # Generate AI response using OpenAI model
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model=model_name
    ).choices[0].message.content

    context.append(f"Customer: {user_query}")
    context.append(f"AI: {response}")
    
    return response

if __name__ == "__main__":
  
    print("Customer Support AI Agent (Type 'exit' to quit)\n")
    print("AI: Hello! How can I assist you today?\n")

    while True:
        user_query = input("You: ") 
        if user_query.lower() == "exit":
            print("Exiting AssistAI...")
            sys.exit()  
        
        response = chat(user_query)
        print("AI:", response)
