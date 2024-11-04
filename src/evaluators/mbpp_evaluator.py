import re
from typing import List, Dict, Any
import evaluate
import torch
from evaluators.base_evaluator import BaseEvaluator
from transformers import StoppingCriteria, StoppingCriteriaList

class MultiSequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stopping_sequences, prompt_length):
        self.tokenizer = tokenizer
        self.stopping_sequences = stopping_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        # Handle batch dimension
        batch_size = input_ids.shape[0]
        should_stop = torch.zeros(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            generated_text = self.tokenizer.decode(input_ids[i][self.prompt_length:])
            should_stop[i] = any(seq in generated_text for seq in self.stopping_sequences)
            
        return should_stop.all()

class MBPPEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer, batch_size: int = 8):
        super().__init__(model, tokenizer, batch_size)
        self.code_eval = evaluate.load('code_eval')
        self.stopping_sequences = ["\n\n#", "\n\nassert", "\n\nprint", "\n\ndef", "\n\nif __name__"]
    
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
        
        # Create stopping criteria for batch
        stopping_criteria = StoppingCriteriaList([
            MultiSequenceStoppingCriteria(
                self.tokenizer, 
                self.stopping_sequences, 
                inputs['input_ids'].shape[1]
            )
        ])

        # Generate completions
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode and clean completions
        completions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return [self.clean_completion(completion) for completion in completions]
    
    def clean_completion(self, completion: str) -> str:
        """Clean up completion by removing trailing code"""
        def_pos = completion.find('def solution')
        marker_pattern = r'\n(#|assert|print|def|if __name__)'
        match = re.search(marker_pattern, completion[def_pos:])
        
        if match:
            return completion[:def_pos + match.start()].rstrip()
        return completion.rstrip()

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