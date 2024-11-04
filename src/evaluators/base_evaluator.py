from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator
from transformers import PreTrainedModel, PreTrainedTokenizer

class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    
    def create_batches(self, dataset: Any) -> Iterator[List[Dict[str, Any]]]:
        """Create batches from dataset"""
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget the last partial batch
            yield batch
    
    @abstractmethod
    def prepare_prompts(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Prepare prompts from a batch of samples"""
        pass
    
    @abstractmethod
    def generate_completions(self, prompts: List[str]) -> List[str]:
        """Generate completions for a batch of prompts"""
        pass
        
    @abstractmethod
    def evaluate_completions(self, completions: List[str], samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of completions against ground truth"""
        pass
    
    @abstractmethod
    def run_evaluation(self, dataset: Any) -> Dict[str, Any]:
        """Run full evaluation on dataset"""
        pass 