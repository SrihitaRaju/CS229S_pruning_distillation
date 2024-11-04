import re
from typing import List, Dict, Any
import evaluate
import torch
from evaluators.base_evaluator import BaseEvaluator


class MBPPEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, batch_size: int = 8):
        super().__init__(model, tokenizer, batch_size)
        self.code_eval = evaluate.load('code_eval')
        self.stopping_sequences = ["\n\n#", "\n\nassert", "\n\nprint", "\n\nif __name__"]
    
    def prepare_prompts(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Prepare MBPP-style prompts for a batch"""
        return [
            f"""# Write a Python function for this task. Only the function, no tests or examples:
# {sample['text']}

def solution""" 
            for sample in samples
        ]

    def generate_completions(self, prompts: List[str]) -> List[str]:
        """Generate completions with MBPP-specific stopping criteria for a batch"""
        # Tokenize all prompts
        self.tokenizer.pad_token = self.tokenizer.eos_token_id
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate completions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode and clean completions
        completions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, truncate_before_pattern=self.stopping_sequences)
        return completions

    def evaluate_completions(self, completions: List[str], samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate completions using MBPP test cases for a batch"""
        results = []
        
        for completion, sample in zip(completions, samples):
            test = sample['test_list'][0]
            func_name = test[7:test.find('(')]
            
            # Replace solution with actual function name
            completion_with_name = completion.replace('def solution', f'def {func_name}')
            
            # Evaluate against test cases
            result = self.code_eval.compute(
                references=['\n'.join(sample['test_list'])],
                predictions=[[completion_with_name]],
                k=[1]
            )
            
            results.append({
                'pass@1': result['pass@1'],
                'completion': completion_with_name,
                'test_results': result[1]
            })
            
        return results

    def run_evaluation(self, dataset) -> Dict[str, Any]:
        """Run evaluation on MBPP dataset using batches"""
        results = []
        total_passed = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.create_batches(dataset)):
            print(f"Processing batch {batch_idx+1}...")
            
            prompts = self.prepare_prompts(batch)
            completions = self.generate_completions(prompts)
            eval_results = self.evaluate_completions(completions, batch)
            
            for i, (sample, eval_result) in enumerate(zip(batch, eval_results)):
                results.append({
                    'sample_id': total_samples + i,
                    'prompt': sample['text'],
                    'completion': eval_result['completion'],
                    'passed': eval_result['pass@1'] == 1.0,
                    'test_results': eval_result['test_results']
                })
                
                total_passed += eval_result['pass@1']
            
            total_samples += len(batch)
        
        return {
            'results': results,
            'total_samples': total_samples,
            'passed_samples': total_passed,
            'pass_rate': total_passed / total_samples
        }