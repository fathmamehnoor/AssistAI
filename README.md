# AssistAI - Customer Support AI Agent

## Overview
This project implements a customer support AI agent that retrieves relevant information from a structured knowledge base and generates responses using an OpenAI model. The system uses ChromaDB for vector storage and retrieval, and SentenceTransformers for embedding text data.

## Approach

### 1. Knowledge Base Construction
- The knowledge base is built from structured text files (`products.txt`, `policies.txt`, `qa.txt`) that contain product details, shipping policies, and FAQs.
- SentenceTransformers (`all-MiniLM-L6-v2`) is used to generate embeddings for text data.
- ChromaDB is used for persistent vector storage, enabling efficient similarity search and retrieval.

### 2. Query Processing & Retrieval
- When a user query is received, it is processed by ChromaDB to find the most relevant knowledge snippet using similarity search.
- The retrieved information is used to provide context-aware responses.

### 3. Response Generation
- The AI agent takes the retrieved knowledge and formulates a response using an OpenAI model (`gpt-4o`).
- The response is generated based on the context of the knowledge and the user's query.

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Virtual environment (optional but recommended)
- Required dependencies

### Installation
```bash
# Clone the repository
git clone https://github.com/fathmamehnoor/AssistAI.git
cd AssistAI
```

```bash
# Create a virtual environment (optional)
python3 -m venv venv # On Windows use `python -m venv venv`
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

```bash
# Install dependencies
pip install chromadb openai sentence-transformers python-dotenv

```

### Environment Variables
Create a `.env` file in the project directory and add the required API credentials:
```
OPENAI_API_KEY="your_api_key"
```

### Running the AI Agent
```bash
python3 knowledge_base.py  # Run this once to set up the knowledge base. On Windows use `python knowledge_base.py`
python3 ai_agent.py  # Start the chatbot. On Windows use `python ai_agent.py`

```
## Example Conversations

Customer Support AI Agent (Type 'exit' to quit)

**You:** How much does the PowerBook Pro cost?  
**AI:** The PowerBook Pro costs $1299. It features a 15.6" Retina display, 512GB SSD storage, 16GB RAM, and an Intel i7 processor. It's available in Space Gray and Silver colors.

**You:** what is the warranty period for SmartWatch Elite? <br>  
**AI:** The SmartWatch Elite comes with a 1-year limited warranty.  

**You:** Is there an option to pay for my orders using installments?  
**AI:** Yes, installment payment plans are available for purchases over $300. This option allows you to pay for your order in smaller, manageable payments. If you'd like to select this payment method, you can do so at checkout.  
