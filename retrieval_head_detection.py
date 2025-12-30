"""
LLM Context Needle Tester for Edgar SEC Filings
Tests incorporation state retrieval from 156_edgar_merged.csv
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import time

class LLMContextNeedleTester:
    def __init__(self,
                 csv_path="156_edgar_merged.csv",
                 model_name="Qwen/Qwen2.5-7B-Instruct",
                 retrieval_question="In which US state was this company incorporated? Answer with only the state name:",
                 results_dir="head_score",
                 device=None):
        """
        Initialize the tester.
        :param csv_path: Path to the CSV file containing filings
        :param model_name: HuggingFace model identifier
        :param retrieval_question: Question to ask the model
        :param results_dir: Directory to save results
        :param device: Device to run model on (cuda/cpu)
        """
        self.csv_path = csv_path
        self.model_name = model_name
        self.retrieval_question = retrieval_question
        self.results_dir = results_dir
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        self.device = 'cuda'
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load tokenizer and model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else "cpu",
            trust_remote_code=True
        )
        self.model.eval()
        self.head_counter = defaultdict(list)
        self.layer_num = len(self.model.model.layers) if hasattr(self.model, 'model') else 0
        self.head_num = self.model.config.num_attention_heads if hasattr(self.model.config, 'num_attention_heads') else 0
        print(f"Loading CSV data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} rows")
        
    def generate_context(self, row):
        section_1 = str(row['section_1']) if pd.notna(row['section_1']) else ""
        return section_1
    
    def extract_first_token(self, response):
        tokens = response.strip().split()
        if tokens:
            return tokens[0].lower()
        return ""
    
    def evaluate_context(self, idx, row):
        inc_state_label = str(row['inc_state']).strip().lower() if pd.notna(row['inc_state']) else ""
        context = self.generate_context(row)
        if not context or len(context.strip()) < 50:
            return {
                'index': idx,
                'inc_state': inc_state_label,
                'score': 0,
                'response': "",
                'first_token': "",
                'match': False
            }
        prompt = [
            {"role": "user", "content": f"<filing>{context}</filing>\nBased on the content of the filing, Question: {self.retrieval_question}\nAnswer:"}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            conversation=prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.device)
        
        # ***get logits directly ***
        try:
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            first_token = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True).strip().lower()
            match = inc_state_label.startswith(first_token.lower()) if inc_state_label else False
            score = 1.0 if match else 0.0
            
            return {
                'index': idx,
                'inc_state': inc_state_label,
                'score': score,
                'response': first_token,
                'first_token': first_token,
                'match': match
            }
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return {
                'index': idx,
                'inc_state': inc_state_label,
                'score': 0,
                'response': str(e),
                'first_token': "",
                'match': False
            }
        
    def run_evaluation(self, max_rows=None):
        """
        Run evaluation on all or subset of contexts.
        
        :param max_rows: Maximum number of rows to evaluate (None for all)
        :return: List of results
        """
        max_rows = max_rows or len(self.df)
        rows_to_eval = min(max_rows, len(self.df))
        
        results = []
        
        print(f"\nEvaluating {rows_to_eval} contexts...")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Question: {self.retrieval_question}")
        print("-" * 80)
        
        start_time = time.time()
        
        for idx in range(rows_to_eval):
            row = self.df.iloc[idx]
            result = self.evaluate_context(idx, row)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Evaluated {idx + 1}/{rows_to_eval} contexts (elapsed: {elapsed:.1f}s)")
        
        elapsed = time.time() - start_time
        print(f"\nTotal evaluation time: {elapsed:.1f}s")
        
        return results
    
    def save_results(self, results):
        """
        Save evaluation results to JSON file.
        
        :param results: List of result dictionaries
        """
        # Summary statistics
        total = len(results)
        matches = sum(1 for r in results if r['match'])
        recall = (matches / total * 100) if total > 0 else 0
        
        summary = {
            'model': self.model_name,
            'question': self.retrieval_question,
            'total_contexts': total,
            'successful_retrievals': matches,
            'recall_percentage': recall,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_data = {
            'summary': summary,
            'results': results
        }
        
        # Save to head_score directory
        model_name_clean = self.model_name.replace('/', '_')
        output_file = os.path.join(self.results_dir, f"{model_name_clean}_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print(f"\nSummary:")
        print(f"  Total Contexts: {total}")
        print(f"  Successful Retrievals: {matches}")
        print(f"  Recall: {recall:.2f}%")
        return output_file


def main():
    """Main entry point for the tester."""
    tester = LLMContextNeedleTester()
    
    # Run evaluation on all contexts
    results = tester.run_evaluation()
    
    # Save results
    tester.save_results(results)
