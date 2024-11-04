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
        if self.tokenizer.pad_token is None:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            padding_side='left'
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

    def evaluate_completions(self, completions: List[str], samples: List[Dict[str, Any]], k: list = [1]) -> List[Dict[str, Any]]:
        """Evaluate completions using MBPP test cases for a batch"""
        results = []

        # Have to add this to allow code_eval to work
        import os
        os.environ['HF_ALLOW_CODE_EVAL'] = '1'
        
        # Extract function names and prepare test cases
        func_names = []
        test_cases = []
        for sample in samples:
            test = sample['test_list'][0]
            func_name = test[7:test.find('(')]
            func_names.append(func_name)
            test_cases.append('\n'.join(sample['test_list']))
        
        # Replace solution with actual function names
        completions_with_names = []
        for completion, func_name in zip(completions, func_names):
            completion_with_name = completion.replace('def solution', f'def {func_name}')
            completions_with_names.append([completion_with_name])
            
        # Evaluate all test cases at once
        results = self.code_eval.compute(
            references=test_cases,
            predictions=completions_with_names,
            k=k
        )
            
        return results, completions_with_names

    def run_evaluation(self, dataset, k: list = [1]) -> Dict[str, Any]:
        """Run evaluation on MBPP dataset using batches"""
        results = []
        total_passed = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.create_batches(dataset)):
            print(f"Processing batch {batch_idx+1}...")
            
            prompts = self.prepare_prompts(batch)
            # Bunch of changes needed to generalize to multiple completions per prompt 
            completions = self.generate_completions(prompts)
            eval_results, completions = self.evaluate_completions(completions, batch, k)
            #import pdb; pdb.set_trace()
            
            #for i, (sample, completion) in enumerate(zip(batch, completions)):
                #results.append({
                #    'sample_id': total_samples + i,
                #    'prompt': sample['text'],
                #    'completion': completion,
                #    'passed': eval_results[1][i][0][1]['passed'] == 1.0,
                #    #'test_results': eval_result['test_results']
                #})
                #total_passed += eval_results[1][i][0][1]['passed'] == 1.0
            results.append(dict(results=eval_results, prompts=prompts,completion=completions))
            
            total_samples += len(batch)
            break
        return {
            'results': results,
            'total_samples': total_samples,
            'passed_samples': total_passed,
            'pass_rate': total_passed / total_samples
        }