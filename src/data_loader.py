import os
import torch
from datasets import load_dataset, load_from_disk, load_dataset_builder, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil
import requests

def load_or_download_llama_model():
    model_path = "models/llama-3.2-1b-instruct"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    if os.path.exists(model_path):
        print("Loading Llama 3.2 1B Instruct model from local storage...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            # Force CPU usage to avoid MPS nonsense 
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map={"": "cpu"})
        except ValueError as e:
            print(f"Error loading model with auto device mapping: {e}")
            print("Attempting to load model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map={"": "cpu"})
        print("Llama 3.2 1B Instruct model loaded successfully.")
    else:
        print("Downloading Llama 3.2 1B Instruct model...")
        
        # Note: You need to have the necessary permissions to download this model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        except ValueError as e:
            print(f"Error downloading model with auto device mapping: {e}")
            print("Attempting to download model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map={"": "cpu"})
        
        print("Llama 3.2 1B Instruct model downloaded successfully.")
        
        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Llama 3.2 1B Instruct model saved locally.")
    
    return tokenizer, model

def load_or_download_coding_dataset():
    dataset_path = "datasets/codeparrot-clean-1000"
    if os.path.exists(dataset_path):
        print("Loading coding dataset from local storage...")
        dataset = load_from_disk(dataset_path)
        print("Coding dataset loaded successfully.")
    else:
        print("Downloading coding dataset (first 1000 samples)...")
        # Check for available disk space
        _, _, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 1:  # Require at least 1GB free space for the subset
            raise IOError(f"Insufficient disk space. {free_gb}GB available, need at least 1GB.")

        dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        dataset = list(dataset.take(1000))
        print("Coding dataset (first 1000 samples) downloaded successfully.")
        
        # Save the dataset
        Dataset.from_list(dataset).save_to_disk(dataset_path)
        print("Coding dataset saved locally.")
    
    print(f"Dataset size: {len(dataset)} samples")
    return dataset



    return None  # Return None if dataset loading fails

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    # Load or download Llama 3.2 1B model
    tokenizer, model = load_or_download_llama_model()

    # Load or download coding dataset
    dataset = load_or_download_coding_dataset()

if __name__ == "__main__":
    main()
