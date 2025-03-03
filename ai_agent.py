import os
import sys
from openai import OpenAI
import chromadb
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Retrieve OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    sys.exit("Error: OPENAI_API_KEY is not set in the environment variables.")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    sys.exit(f"Error initializing OpenAI client: {e}")

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_collection(name="support_knowledge")

# Store previous interactions for session memory
context = []

def retrieve_knowledge(query: str) -> str:
    """
    Queries ChromaDB for relevant knowledge based on user input.

    Args:
        query (str): The user query.

    Returns:
        str: The most relevant knowledge snippet or an error message if no data is found.
    """
    results = collection.query(query_texts=[query], n_results=1)
    return results["documents"][0] if results["documents"] else "No relevant information found."


def chat(user_query: str, context: list) -> str:
    """
    Generates a response using the retrieved knowledge and OpenAI API.

    Args:
        user_query (str): The user input query.
        context (list): Previous conversation history for maintaining context.

    Returns:
        str: AI-generated response.
    """
    knowledge = retrieve_knowledge(user_query)
    conversation_history = "\n".join(context[-4:])  # Keep last 3 interactions

    prompt = (
        f"You are a helpful customer support AI. Answer the customer's query using this knowledge:\n\n"
        f"{knowledge}\n\n"
        f"Here is the recent conversation history to maintain context:\n"
        f"{conversation_history}\n"
        f"Customer: {user_query}\n"
        f"AI:"
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="gpt-4o"
        ).choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

    context.append(f"Customer: {user_query}")
    context.append(f"AI: {response}")

    return response


def main():
    """
    Main function to run the interactive CLI chatbot.
    """
    print("Customer Support AI Agent (Type 'exit' to quit)\n")
    print("AI: Hello! How can I assist you today?\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() == "exit":
            print("Exiting AssistAI...")
            sys.exit()

        response = chat(user_query, context)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
