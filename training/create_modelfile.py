import json

def create_training_data():
    """Create training data in Ollama format"""
    
    # Convert your existing knowledge base to training format
    training_examples = [
    {
        "prompt": "What are your express delivery options?",
        "response": "Express shipping is available and takes 1-2 business days. It costs $12.99."
    },
    {
        "prompt": "Is there any shipping fee waived?",
        "response": "Yes, we offer free standard shipping on orders over $50."
    },
    {
        "prompt": "Can I return something without the original box?",
        "response": "Returns are accepted only for unused items in their original packaging. Without the original packaging, the return may be denied."
    },
    {
        "prompt": "What's your return policy for defective electronics?",
        "response": "Defective products can be returned within 30 days, and return shipping is free."
    },
    {
        "prompt": "Do you support EMI or split payments?",
        "response": "Yes, installment payment plans are available for purchases over $300."
    },
    {
        "prompt": "What warranty do your electronics come with?",
        "response": "All electronics include the manufacturer's warranty. You can also purchase extended protection plans at checkout."
    },
    {
        "prompt": "Do you offer international delivery?",
        "response": "Yes, we ship internationally to select countries for an extra fee."
    },
    {
        "prompt": "Can I use a digital wallet to pay?",
        "response": "Yes, we accept the PaySecure digital wallet in addition to major credit cards and gift cards."
    },
    {
        "prompt": "Do you ship to military addresses or P.O. boxes?",
        "response": "No, we currently do not ship to P.O. boxes or military addresses."
    },
    {
        "prompt": "What happens if I damage my device accidentally?",
        "response": "Unfortunately, accidental or water damage is not covered under our standard warranty."
    },
    {
        "prompt": "Are gift cards refundable?",
        "response": "Gift cards are non-refundable, but they can be used toward any purchase."
    },
    {
        "prompt": "What personal data do you store?",
        "response": "We store only necessary information and use industry-standard encryption. You can request deletion of your data at any time."
    },
    {
        "prompt": "Can I unsubscribe from marketing emails?",
        "response": "Yes, all marketing communications require opt-in, and you can unsubscribe at any time."
    },
    {
        "prompt": "What if I modify the product myself? Is the warranty still valid?",
        "response": "Modifying or repairing a product by unauthorized persons voids the warranty."
    },
    {
        "prompt": "How long does SmartWatch Elite's battery last?",
        "response": "The SmartWatch Elite offers up to 2-day battery life and includes GPS and heart rate monitoring."
    },
    {
        "prompt": "How much storage does the UltraPhone X come with?",
        "response": "The UltraPhone X comes with 128GB of storage."
    },
    {
        "prompt": "Can I return a special order item?",
        "response": "No, special order items are non-returnable."
    },
    {
        "prompt": "Is the PowerBook Pro still available?",
        "response": "Yes, the PowerBook Pro is currently in stock."
    },
    {
        "prompt": "Can I pay using cash or check?",
        "response": "No, we do not accept checks or money orders."
    },
    {
        "prompt": "Are your products water resistant?",
        "response": "The SmartWatch Elite is water resistant, but not all products are. Please refer to the individual product details."
    },
    {
        "prompt": "I received the wrong item. What can I do?",
        "response": "Please contact support. If the item is incorrect, we will arrange free return shipping and a replacement."
    },
    {
        "prompt": "Can I request store credit instead of a refund?",
        "response": "Yes, if you return an item with a gift receipt, you'll receive store credit."
    },
    {
        "prompt": "Is my info sold to advertisers?",
        "response": "No, we never sell customer data to third parties."
    },
    {
        "prompt": "How long does it take to process a return?",
        "response": "Returns are typically processed within 5–7 business days after receiving the item."
    },
    {
        "prompt": "Do you accept crypto payments?",
        "response": "Currently, we do not support cryptocurrency payments."
    },
    {
        "prompt": "Is the PowerBook Pro good for video editing?",
        "response": "Yes, with a 15.6\" Retina display, 512GB SSD, and 16GB RAM, it's well-suited for video editing."
    },
    {
        "prompt": "What's the camera quality like on the UltraPhone X?",
        "response": "The UltraPhone X features a 12MP camera with Face Recognition and a 6.5\" OLED display."
    },
    {
        "prompt": "How do I know if my country is eligible for international shipping?",
        "response": "During checkout, eligible countries will be listed if international shipping is available."
    },
    {
        "prompt": "How do I track my shipment?",
        "response": "Once your order ships, you'll receive a tracking number via email or SMS."
    },
    {
        "prompt": "Can I upgrade shipping after placing an order?",
        "response": "Please contact support immediately. If the order hasn't shipped yet, we may be able to upgrade your shipping."
    }
    ]
    
    # Save as training file
    with open("training_data.txt", "w") as f:
        for example in training_examples:
            f.write(f"### Human: {example['prompt']}\n\n")
            f.write(f"### Assistant: {example['response']}\n\n")
    
    print(f"Created training data with {len(training_examples)} examples")

def create_modelfile(base_model="llama2:7b-chat", model_name="customer-support"):
    """Create Ollama Modelfile for fine-tuning"""
    
    modelfile_content = f"""FROM {base_model}

# Set custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# System prompt for customer support
SYSTEM You are a helpful customer support assistant. Answer questions based on company policies and be professional and friendly.

# Load training data
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile for {model_name}")
    return model_name

if __name__ == "__main__":
    create_training_data()
    model_name = create_modelfile()
    
    print(f"\nTo create your fine-tuned model, run:")
    print(f"ollama create {model_name} -f Modelfile")