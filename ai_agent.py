import os
import sys
import chromadb
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path

# Add local_llm folder to path
sys.path.append(str(Path(__file__).parent / "local_llm"))

# Import Ollama wrapper
from local_llm.ollama_wrapper import OllamaWrapper

# Load environment variables
load_dotenv()

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_collection(name="support_knowledge")

# Store previous interactions for session memory
context = []

# Initialize Ollama wrapper
ollama = OllamaWrapper(
    base_model="llama2:7b-chat",
    fine_tuned_model="customer-support:latest"  # Change if your model is named differently
)

def retrieve_knowledge(query: str) -> str:
    """
    Queries ChromaDB for relevant knowledge based on user input.
    """
    results = collection.query(query_texts=[query], n_results=1)
    return results["documents"][0] if results["documents"] else "No relevant information found."

def chat_response(user_query: str, history: list) -> tuple:
    """
    Generates a response using the retrieved knowledge and Ollama model.
    Returns updated history for Gradio chatbot interface.
    """
    global context
    
    knowledge = retrieve_knowledge(user_query)
    conversation_history = "\n".join(context[-4:])  # Last 4 messages max
    
    # Build context for Ollama
    full_context = f"{knowledge}\n\nConversation History:\n{conversation_history}"
    
    result = ollama.generate_response(
        query=user_query,
        context=full_context,
        use_fine_tuned=True
    )
    
    response = result["response"]
    
    # Update session memory
    context.append(f"Customer: {user_query}")
    context.append(f"AI: {response}")
    
    # Update chat history for Gradio (using messages format)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response})
    
    return history, ""

def clear_chat():
    """
    Clears the chat history and context.
    """
    global context
    context = []
    return [], ""

def get_system_status():
    """
    Returns system status information.
    """
    try:
        # Check if collection exists and has documents
        collection_count = collection.count()
        ollama_status = "Connected" if ollama else "Not Connected"
        
        status = f"""
        **System Status:**
        - ChromaDB Collection: {collection_count} documents
        - Ollama Status: {ollama_status}
        - Base Model: {ollama.base_model if ollama else 'N/A'}
        - Fine-tuned Model: {ollama.fine_tuned_model if ollama else 'N/A'}
        """
        return status
    except Exception as e:
        return f"Error getting system status: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="Customer Support AI Agent",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 20px !important;
    margin: 0 auto !important;
    }

    .chat-container {
        height: 600px !important;
    }
    #chatbot {
        border-radius: 12px !important;
    }
    .center-title {
        text-align: center !important;
    }
    .center-subtitle {
        text-align: center !important;
        color: #666 !important;
    }
    """
) as demo:
    
    gr.Markdown(
        "# 🤖 Customer Support AI Agent", 
        elem_classes="center-title"
    )
    gr.Markdown(
        "Welcome! I'm here to help you with your questions. Ask me anything!", 
        elem_classes="center-subtitle"
    )
    
    
    with gr.Row():
        # Left spacer (smaller)
        with gr.Column(scale=1, min_width=50):
            pass
            
        # Main chat interface (wider)
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                value=[],
                elem_id="chatbot",
                height=600,
                show_label=False,
                container=True,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
                
        # Right sidebar (smaller)
        with gr.Column(scale=2, min_width=200):
            # System information panel
            gr.Markdown("### 📊 System Information")
            
            status_display = gr.Markdown(get_system_status())
            
            refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
            
            gr.Markdown("### 📈 Session Stats")
            session_stats = gr.Markdown("**Messages in Context:** 0")
            
            # Knowledge base info
            gr.Markdown("### 📚 Knowledge Base")
            kb_info = gr.Markdown(f"**Documents:** {collection.count()}")
            
    # Event handlers
    def update_stats():
        return f"**Messages in Context:** {len(context)}"
    
    def submit_message(user_message, history):
        if user_message.strip():
            new_history, empty_textbox = chat_response(user_message, history)
            stats = update_stats()
            return new_history, empty_textbox, stats
        return history, user_message, update_stats()
    
    # Button click events
    submit_btn.click(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, session_stats]
    )
    
    # Enter key press event
    msg.submit(
        submit_message,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, session_stats]
    )
    
    # Clear chat event
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    ).then(
        lambda: update_stats(),
        outputs=[session_stats]
    )
    
    # Refresh status event
    refresh_status_btn.click(
        get_system_status,
        outputs=[status_display]
    )

def main():
    """
    Main function to launch the Gradio interface.
    """
    print("Starting Customer Support AI Agent Web Interface...")
    print("ChromaDB initialized with collection:", collection.name)
    print("Ollama wrapper initialized")
    
    # Launch Gradio interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        show_error=True,
        inbrowser=True          # Automatically open browser
    )

if __name__ == "__main__":
    main()