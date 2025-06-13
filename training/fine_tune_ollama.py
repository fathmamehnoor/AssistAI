# training/fine_tune_ollama.py
import subprocess
import ollama
import os

class OllamaFineTuner:
    def __init__(self, base_model="llama2:7b-chat"):
        self.base_model = base_model
        self.client = ollama.Client()
    
    def create_fine_tuned_model(self, model_name="customer-support"):
        """Create fine-tuned model using Ollama"""
        
        try:
            # Run the create command
            result = subprocess.run([
                "ollama", "create", model_name, "-f", "Modelfile"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully created model: {model_name}")
                return model_name
            else:
                print(f"Error creating model: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Exception during model creation: {e}")
            return None
    
    def test_fine_tuned_model(self, model_name):
        """Test the fine-tuned model"""
        
        test_queries = [
            "What is your return policy?",
            "Do you offer free shipping?", 
            "How can I get technical support?"
        ]
        
        print(f"\n Testing fine-tuned model: {model_name}")
        print("=" * 50)
        
        for query in test_queries:
            try:
                response = self.client.generate(
                    model=model_name,
                    prompt=query,
                    options={'temperature': 0.7}
                )
                
                print(f"\nQ: {query}")
                print(f"A: {response['response']}")
                print("-" * 30)
                
            except Exception as e:
                print(f"Error testing query '{query}': {e}")

if __name__ == "__main__":
    fine_tuner = OllamaFineTuner()
    
    # Create the fine-tuned model
    model_name = fine_tuner.create_fine_tuned_model("customer-support")
    
    if model_name:
        # Test the model
        fine_tuner.test_fine_tuned_model(model_name)
        
        print(f"\nFine-tuning complete!")
        print(f"Use model name: {model_name}")
    else:
        print("Fine-tuning failed")