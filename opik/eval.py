# -----------------------------
# 1. Import dependencies
# -----------------------------
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import hashlib
import numpy as np
from typing import Any, Dict

from opik import Opik, track
from opik.evaluation import evaluate
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation import models

# -----------------------------
# 2. Configuration Variables
# -----------------------------
# API Configuration
api_url = "http://127.0.0.1:8000/host/agentic-rag/chat/"

# Model Configuration
llm_model_name = "gpt-5-mini"
model_name = llm_model_name

# Dataset Configuration
should_create_dataset = True
csv_url = "https://docs.google.com/spreadsheets/d/1tgpXeOjpEo__a8SER9EQ1VLIAHlB3atzPZevauPcMnM/export?format=csv"
dataset_name = "travela_eval_v4"
project_tracing_name = "Travela Evaluation API"

# Evaluation Configuration
should_evaluate_dataset = True
experiment_name = llm_model_name + "-" + "outcomes"

# Correctness evaluation prompt - Make it clearer
CORRECTNESS_PROMPT = """
You are a helpful assistant for evaluating the correctness of an output.

Given a user query, ground truth, and the model's response, please assess the correctness of the output. The model's response may be in English, Banglish (Bangla written in English script), or Bangla.

Assign a binary rating based on whether the model's response reasonably aligns with the ground truth. Minor variations in phrasing or language are acceptable as long as the core meaning and content is correct. Also ignore the emojis or new line or special characters in the model response.

Respond with ONLY the number:
- 1 for correct
- 0 for incorrect

User Query: {user_query}

Ground Truth: {ground_truth}

Model Response: {model_response}
"""

# -----------------------------
# 3. Connect to Opik
# -----------------------------
load_dotenv()
client = Opik(project_name="Travela Evaluation Test")

# -----------------------------
# 4. Custom Correctness Evaluator
# -----------------------------
# Fix the LiteLLMChatModel class name (around line 66)
class CorrectnessEvaluator(base_metric.BaseMetric):
    def __init__(self, name: str = "correctness_evaluator", model_name: str = "gpt-5-mini"):
        self.name = name
        self.model_name = model_name
        # Use Opik's LiteLLMChatModel for better LLM provider support
        self.llm_client = models.LiteLLMChatModel(
            model_name=model_name,
            temperature= 1 if model_name == "gpt-5-mini" else 0
        )
    
    def score(self, output: str, reference: str, input: str, **ignored_kwargs: Any):
        """
        Score the correctness of the output against the reference.
        
        Args:
            output: The model's response
            reference: The ground truth/expected output
            input: The user query/question
            **ignored_kwargs: Additional arguments (ignored)
        """
        try:
            final_prompt = CORRECTNESS_PROMPT.format(
                user_query=input,
                ground_truth=reference,
                model_response=output
            )
            
            # Use generate_provider_response instead of generate_chat_completion
            response = self.llm_client.generate_provider_response(
                messages=[{"role": "user", "content": final_prompt}]
            )
            
            # Extract the score from response
            response_content = response.choices[0].message.content.strip()
            
            # Handle various response formats
            if response_content in ['0', '1']:
                score_value = int(response_content)
            else:
                # Try to extract first digit if response is longer
                for char in response_content:
                    if char in ['0', '1']:
                        score_value = int(char)
                        break
                else:
                    score_value = 0  # Default to 0 if no valid score found
            
            return score_result.ScoreResult(
                name=self.name,
                value=score_value,
                reason=f"Correctness evaluation: {score_value} (Response: {response_content})"
            )
            
        except Exception as e:
            print(f"Error in correctness evaluation: {e}")
            return score_result.ScoreResult(
                name=self.name,
                value=0,
                reason=f"Error during evaluation: {str(e)}"
            )

# -----------------------------
# 5. Token and Cost Metrics
# -----------------------------
class TokensMetric(base_metric.BaseMetric):
    def __init__(self, name: str = "tokens"):
        self.name = name
    
    def score(self, tokens_used: int = 0, **ignored_kwargs: Any):
        return score_result.ScoreResult(
            name=self.name,
            value=tokens_used,
            reason=f"Tokens used: {tokens_used}"
        )

class CostMetric(base_metric.BaseMetric):
    def __init__(self, name: str = "cost"):
        self.name = name
    
    def score(self, cost: float = 0.0, **ignored_kwargs: Any):
        return score_result.ScoreResult(
            name=self.name,
            value=float(cost),
            reason=f"Cost: ${cost}"
        )

# -----------------------------
# 6. Create or load dataset
# -----------------------------
# Fix dataset duplicate detection logic
if should_create_dataset:
    print("Loading dataset from CSV...")
    df = pd.read_csv(csv_url)
    
    # Get or create dataset
    dataset = client.get_or_create_dataset(name=dataset_name)
    
    # Function to create a unique identifier for each example
    def create_example_hash(property_id, question, answer):
        """Create a unique hash for an example based on property_id, question, and answer"""
        content = f"{property_id}|{question}|{answer}"
        return hashlib.md5(content.encode()).hexdigest()
    
    # Check existing examples to avoid duplicates
    try:
        existing_items = dataset.get_items()  # Fix: Use get_items() instead of list_items()
        existing_hashes = set()
        for item in existing_items:
            # Fix: Update to match your dataset structure (Property_ID, Input, Output)
            prop_id = item.get('Property_ID', '') if 'Property_ID' in item else ''
            question = item.get('Input', '') if 'Input' in item else ''
            answer = item.get('Output', '') if 'Output' in item else ''
            example_hash = create_example_hash(prop_id, question, answer)
            existing_hashes.add(example_hash)
        print(f"Found {len(existing_hashes)} existing examples in dataset.")
    except Exception as e:
        existing_hashes = set()
        print(f"No existing examples found or error checking existing data: {e}")

    # Prepare new examples from CSV
    new_examples = []
    updated_count = 0
    skipped_count = 0
    
    for index, row in df.iterrows():
        property_id = str(row["Property_ID"])
        question = str(row["Input"])
        answer = str(row["Output"])
        
        # Create hash for this example
        example_hash = create_example_hash(property_id, question, answer)
        
        # Check if this example already exists
        if example_hash not in existing_hashes:
            new_examples.append({
                "Property_ID": property_id,
                "Input": question,
                "Output": answer
            })
            updated_count += 1
        else:
            skipped_count += 1
    
    # Add new examples to the dataset
    if new_examples:
        try:
            dataset.insert(new_examples)
            print(f"Successfully added {len(new_examples)} new examples to dataset '{dataset_name}'.")
            print(f"Summary: {updated_count} new, {skipped_count} skipped (already exist)")
        except Exception as e:
            print(f"Error adding examples to dataset: {e}")
            raise
    else:
        print(f"No new examples to add. All {len(df)} examples from CSV already exist in the dataset.")
else:
    # Just get the existing dataset
    dataset = client.get_or_create_dataset(name=dataset_name)

# -----------------------------
# 7. Define the evaluation task
# -----------------------------
@track  # Logs traces automatically into Opik
def get_target_output(inputs: dict, metadata: dict) -> dict:
    """Call the API and return the response"""
    print(f"Inputs: {inputs}")
    print(f"Metadata: {metadata}")
    
    try:
        response = requests.post(
            api_url,
            headers={
                'Content-Type': 'application/json',
            },
            data=json.dumps({
                'property_id': str(metadata.get('property_id', inputs.get('property_id', ''))),
                'question': inputs['question'],
            })
        )
        
        resp = response.json()
        print(f"Response: {resp}")
        
        # Extract chat_response
        chat_response = resp.get("data", {}).get("chat_response", {})
        
        return {
            "output": chat_response.get("answer", ""),
            "tokens_used": chat_response.get("tokens_used", 0),
            "cost": chat_response.get("cost", 0.0),
            "session_id": chat_response.get("session_id", ""),
        }
    except Exception as e:
        print(f"Error calling API: {e}")
        return {
            "output": "Error occurred",
            "tokens_used": 0,
            "cost": 0.0,
            "session_id": "",
        }

# Fix the evaluation_task function (around line 269)
@track
def evaluation_task(item):
    """Takes dataset item and runs pipeline."""
    try:
        # Fix: Access the correct keys based on your dataset structure
        # Your dataset has Property_ID, Input, Output columns
        property_id = item.get("Property_ID", "")
        question = item.get("Input", "")  # Use "Input" instead of "question"
        expected_answer = item.get("Output", "")  # Use "Output" instead of "expected_output"
        
        # Prepare inputs for the API call
        inputs = {"question": question}  # API expects "question" key
        metadata = {"property_id": property_id}
        
        # Get the API response
        result = get_target_output(inputs, metadata)
        
        # Return the result with proper keys for metrics
        return {
            "output": result["output"],
            "tokens_used": result["tokens_used"],
            "cost": result["cost"],
            "session_id": result["session_id"],
            "input": question,  # Use the actual question from dataset
            "reference": expected_answer  # Use the actual expected answer from dataset
        }
    except Exception as e:
        print(f"Error in evaluation_task: {e}")
        return {
            "output": "Error occurred",
            "tokens_used": 0,
            "cost": 0.0,
            "session_id": "",
            "input": item.get("Input", ""),
            "reference": item.get("Output", "")
        }

# -----------------------------
# 8. Run evaluation
# -----------------------------
# Fix the evaluation setup to include project name in experiment config
if should_evaluate_dataset:
    print("Starting evaluation...")
    
    # Define the metrics
    correctness_metric = CorrectnessEvaluator(model_name=model_name)
    tokens_metric = TokensMetric()
    cost_metric = CostMetric()
    
    evaluation = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=[correctness_metric, tokens_metric, cost_metric],
        experiment_name=experiment_name,
        experiment_config={
            "project_name": "Travela Evaluation Test",
            "pipeline": "travela_rag",
            "model": llm_model_name,
            "api_url": api_url,
            "dataset_source": csv_url
        },
        verbose=1,
        task_threads=1  # Use single thread to avoid API rate limits
    )
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETED ✅")
    print("="*50)
    print("Results saved in Opik dashboard.")
    print(f"Experiment name: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    
    # Print basic statistics if available
    try:
        print(f"\nEvaluation result type: {type(evaluation)}")
        if hasattr(evaluation, 'experiment_name'):
            print(f"Experiment: {evaluation.experiment_name}")
        
        # Check available attributes
        available_attrs = [attr for attr in dir(evaluation) if not attr.startswith('_')]
        print(f"Available attributes: {available_attrs}")
        
    except Exception as e:
        print(f"Note: Could not extract detailed statistics: {e}")
    
    print("\nView detailed results and analytics in your Opik dashboard!")
    print("Dashboard URL: https://www.comet.com/opik (or your self-hosted instance)")
else:
    print("Evaluation skipped (should_evaluate_dataset = False)")

print("\nScript completed successfully! 🎉")