# local_llm/ollama_wrapper.py
import ollama
import requests
import json
import time

class OllamaWrapper:
    def __init__(self, base_model="llama2:7b-chat", fine_tuned_model=None):
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model or base_model
        self.client = ollama.Client()
        
        # Check if models are available
        self._check_models()
    
    def _check_models(self):
    
        """Check if required models are installed"""
        try:
            response = self.client.list()
            models = response.models  # Access the models attribute
            
            # Extract model names using the 'model' attribute
            model_names = [model.model for model in models]
            
            print(f"Available models: {model_names}")
            
            if self.base_model not in model_names:
                print(f"Pulling {self.base_model}...")
                self.client.pull(self.base_model)
                
            if self.fine_tuned_model not in model_names and self.fine_tuned_model != self.base_model:
                print(f"Fine-tuned model {self.fine_tuned_model} not found. Using base model.")
                self.fine_tuned_model = self.base_model
                
        except Exception as e:
            print(f"Error checking models: {e}")
    
    def generate_response(self, query, context=None, use_fine_tuned=True):
        """Generate response using Ollama"""
        
        # Choose model
        model = self.fine_tuned_model if use_fine_tuned else self.base_model
        
        # Create prompt
        if context:
            prompt = f"""Based on the following context, answer the customer's question professionally:

Context: {' '.join(context) if isinstance(context, list) else context}

Customer Question: {query}

Answer:"""
        else:
            prompt = f"Customer Question: {query}\n\nAnswer:"
        
        try:
            start_time = time.time()
            
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': 200,
                    'stop': ['Customer:', 'Context:']
                }
            )
            
            end_time = time.time()
            
            return {
                'response': response['response'].strip(),
                'model_used': model,
                'processing_time': round(end_time - start_time, 2)
            }
            
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'model_used': model,
                'processing_time': 0
            }
    
    def compare_models(self, query, context=None):
        """Compare base model vs fine-tuned model"""
        base_result = self.generate_response(query, context, use_fine_tuned=False)
        fine_tuned_result = self.generate_response(query, context, use_fine_tuned=True)
        
        return {
            'base_model': {
                'response': base_result['response'],
                'time': base_result['processing_time']
            },
            'fine_tuned': {
                'response': fine_tuned_result['response'], 
                'time': fine_tuned_result['processing_time']
            }
        }