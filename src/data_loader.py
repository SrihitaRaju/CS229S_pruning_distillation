import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_llama_model():
    print("Downloading Llama 2 7B model...")
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # Note: You need to have the necessary permissions to download this model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Llama 2 7B model downloaded successfully.")
    return tokenizer, model

def download_coding_dataset():
    print("Downloading coding dataset...")
    # You can replace this with any coding dataset available on Hugging Face
    dataset = load_dataset("codeparrot/codeparrot-clean-valid")
    print("Coding dataset downloaded successfully.")
    return dataset

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    # Download Llama 2 7B model
    tokenizer, model = download_llama_model()
    
    # Save the model and tokenizer
    model.save_pretrained("models/llama-2-7b")
    tokenizer.save_pretrained("models/llama-2-7b")

    # Download coding dataset
    dataset = download_coding_dataset()
    
    # Save the dataset
    dataset.save_to_disk("datasets/codeparrot-clean-valid")

if __name__ == "__main__":
    main()
