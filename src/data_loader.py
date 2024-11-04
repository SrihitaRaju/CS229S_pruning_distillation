import os
import torch
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil

class ModelLoader:
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)
        
    def load_model(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        """Load or download a model based on its name."""
        model_path = os.path.join("models", model_name.split('/')[-1].lower())
        
        if os.path.exists(model_path):
            print(f"Loading {model_name} from local storage...")
            return self._load_local_model(model_path)
        else:
            print(f"Downloading {model_name}...")
            return self._download_model(model_name, model_path)
    
    def _load_local_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                device_map="auto"
            )
        except ValueError as e:
            print(f"Error loading model with auto device mapping: {e}")
            print("Attempting to load model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map={"": "cpu"}
            )
        return tokenizer, model
    
    def _download_model(self, model_name, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        except ValueError as e:
            print(f"Error downloading model with auto device mapping: {e}")
            print("Attempting to download model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map={"": "cpu"}
            )
        
        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved locally to {model_path}")
        
        return tokenizer, model

    def load_dataset(self, dataset_name, split="train", sample_size=None):
        """Load or download a dataset based on its name and split."""
        dataset_path = os.path.join("datasets", dataset_name.replace('/', '-'), split)
        
        if os.path.exists(dataset_path):
            print(f"Loading {dataset_name} ({split} split) from local storage...")
            dataset = load_from_disk(dataset_path)
        else:
            print(f"Downloading {dataset_name} ({split} split)...")
            dataset = self._download_dataset(dataset_name, split, sample_size)
            
        print(f"Dataset size: {len(dataset)} samples")
        return dataset
    
    def _download_dataset(self, dataset_name, split, sample_size):
        _, _, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 1:
            raise IOError(f"Insufficient disk space. {free_gb}GB available, need at least 1GB.")
            
        dataset = hf_load_dataset(
            dataset_name,
            split=split,
        )
        if sample_size:
            dataset = list(dataset.take(sample_size))
        
        save_path = os.path.join("datasets", dataset_name.replace('/', '-'), split)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Dataset.from_list(dataset).save_to_disk(save_path)
        print(f"Dataset saved locally to {save_path}")
        
        return Dataset.from_list(dataset)

def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    loader = ModelLoader()
    return loader.load_model(model_name)

def load_dataset(dataset_name, split="train", sample_size=None):
    loader = ModelLoader()
    return loader.load_dataset(dataset_name, split, sample_size)

def main():
    # Load or download model
    tokenizer, model = load_model()
    
    # Load or download dataset
    dataset = hf_load_dataset("openai/humaneval", split="test")

if __name__ == "__main__":
    main()
