import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_and_save_model():
    # Define model and tokenizer
    model_name = "microsoft/Phi-3.5-mini-instruct"
    model_dir = "models/"
    
    # Clear existing model directory to avoid cache issues
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    # Optimize for CPU with float16 and eager attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

if __name__ == "__main__":
    load_and_save_model()