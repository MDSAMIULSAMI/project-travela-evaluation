"""
Travela Evaluation Script

This script evaluates the Travela API using LangSmith for dataset creation,
evaluation, and performance analysis.

CSV Structure:
- Property_ID: The property_id
- Input: Contains the input data
- Output: Contains the corresponding expected output
"""

import requests
import json
import pandas as pd
from langsmith import Client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import hashlib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = "http://127.0.0.1:8000/host/agentic-rag/chat/"

# Model configuration
# LLM_MODEL_NAME = "gpt-4o-mini"
# LLM_MODEL_NAME = "gpt-4o"
# LLM_MODEL_NAME = "gpt-4.1-nano"
# LLM_MODEL_NAME = "gpt-4.1"
LLM_MODEL_NAME = "gpt-4.1-mini"
# LLM_MODEL_NAME = "gpt-5"
# LLM_MODEL_NAME = "gpt-5-nano"
# LLM_MODEL_NAME = "gpt-5-mini"

# Dataset configuration
SHOULD_CREATE_DATASET = True
CSV_URL = "https://docs.google.com/spreadsheets/d/1tgpXeOjpEo__a8SER9EQ1VLIAHlB3atzPZevauPcMnM/export?format=csv"
DATASET_NAME = "travela_eval_python_script_v1"
PROJECT_TRACING_NAME = "Travela Evaluation API"

# Evaluation configuration
SHOULD_EVALUATE_DATASET = True
EXPERIMENT_NAME = f"{LLM_MODEL_NAME}-outcomes"
MODEL_NAME = LLM_MODEL_NAME

# Correctness evaluation prompt
CORRECTNESS_PROMPT = """
You are a helpful assistant for evaluating the correctness of an output.

Given a user query, ground truth, and the model's response, please assess the correctness of the output. The model's response may be in English, Banglish (Bangla written in English script), or Bangla.

Assign a binary rating (1 for correct, 0 for incorrect) based on whether the model's response reasonably aligns with the ground truth. But Note that no explanation required at all. Minor variations in phrasing or language are acceptable as long as the core meaning and content is correct. Also ignore the imojies or new line or spacial characters in the model response.

User Query: {user_query}

Ground Truth: {ground_truth}

Model Response: {model_response}
"""


def create_example_hash(property_id, question, answer):
    """Create a unique hash for an example based on property_id, question, and answer"""
    content = f"{property_id}|{question}|{answer}"
    return hashlib.md5(content.encode()).hexdigest()


def create_dataset(client, csv_url, dataset_name):
    """Create or update the LangSmith dataset from CSV data"""
    if not SHOULD_CREATE_DATASET:
        return
        
    df = pd.read_csv(csv_url)
    
    # Check if dataset already exists
    dataset_exists = False
    existing_examples = []
    
    try:
        if client.has_dataset(dataset_name=dataset_name):
            print(f"Dataset '{dataset_name}' already exists. Checking for updates...")
            dataset = client.read_dataset(dataset_name=dataset_name)
            dataset_exists = True
            
            # Get existing examples from LangSmith
            existing_examples = list(client.list_examples(dataset_name=dataset_name))
            print(f"Found {len(existing_examples)} existing examples in dataset.")
        else:
            # Create new dataset
            print(f"Creating new dataset: {dataset_name}")
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Dataset for evaluation",
            )
            print(f"Dataset '{dataset_name}' created successfully.")
    except Exception as e:
        print(f"Error checking/creating dataset: {e}")
        raise
    
    # Create a set of existing example hashes for quick lookup
    existing_hashes = set()
    if dataset_exists:
        for example in existing_examples:
            prop_id = example.inputs.get('property_id', '')
            question = example.inputs.get('question', '')
            answer = example.outputs.get('ground_truth', '')
            example_hash = create_example_hash(prop_id, question, answer)
            existing_hashes.add(example_hash)
    
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
                "inputs": {
                    "question": question,
                    "property_id": property_id,
                },
                "outputs": {
                    "ground_truth": answer,
                },
                "metadata": {
                    "property_id": property_id,
                    "example_hash": example_hash,
                    "source": "csv_import",
                    "import_timestamp": pd.Timestamp.now().isoformat()
                }
            })
            updated_count += 1
        else:
            skipped_count += 1
    
    # Add new examples to the dataset
    if new_examples:
        try:
            client.create_examples(
                dataset_id=dataset.id,
                examples=new_examples
            )
            print(f"Successfully added {len(new_examples)} new examples to dataset '{dataset_name}'.")
            print(f"Summary: {updated_count} new, {skipped_count} skipped (already exist)")
        except Exception as e:
            print(f"Error adding examples to dataset: {e}")
            raise
    else:
        print(f"No new examples to add. All {len(df)} examples from CSV already exist in the dataset.")
    
    # Show dataset statistics
    try:
        final_examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"Dataset '{dataset_name}' now contains {len(final_examples)} total examples.")
    except Exception as e:
        print(f"Warning: Could not retrieve final dataset statistics: {e}")


def get_target_output(inputs: dict, metadata: dict) -> dict:
    """Get the target output from the Travela API"""
    print(f"Inputs: {inputs}")
    print(f"Metadata: {metadata}")
    
    response = requests.post(
        API_URL,
        headers={
            'Content-Type': 'application/json',
        },
        data=json.dumps({
            'property_id': str(metadata['property_id']),
            'question': inputs['question'],
        })
    )
    
    resp = response.json()
    print(f"Response: {resp}")
    
    # Extract chat_response
    chat_response = resp.get("data", {}).get("chat_response", {})
    
    return {
        "answer": chat_response.get("answer", ""),
        "tokens_used": chat_response.get("tokens_used", 0),
        "cost": chat_response.get("cost", 0.0),
        "session_id": chat_response.get("session_id", ""),
    }


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate the correctness of the model output"""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    final_prompt = CORRECTNESS_PROMPT.format(
        user_query=inputs['question'],
        ground_truth=reference_outputs['ground_truth'],
        model_response=outputs['answer']
    )

    response = llm.invoke(final_prompt)

    return int(response.content.strip())


def tokens(inputs: dict, outputs: dict, reference_outputs: dict):
    """Extract token count from outputs"""
    return outputs.get('tokens_used', 0)


def cost(inputs: dict, outputs: dict, reference_outputs: dict):
    """Extract cost from outputs"""
    return str(outputs.get('cost', 0.0))


def process_correctness_score(score):
    """Process correctness score to ensure it's a valid binary value"""
    if pd.isna(score):
        return 0
    if isinstance(score, str):
        # If it's a string of 1s and 0s, take the first character
        if score and score[0] in ['0', '1']:
            return int(score[0])
        # Try to convert to int/float
        try:
            return int(float(score))
        except:
            return 0
    try:
        return int(float(score))
    except:
        return 0


def analyze_results(result):
    """Analyze and display evaluation results"""
    # Convert result to DataFrame for easier analysis
    df_results = result.to_pandas()
    df_results['correctness_numeric'] = df_results['feedback.correctness_evaluator'].apply(process_correctness_score)
    
    # Basic Statistics
    print("\nOVERALL PERFORMANCE METRICS")
    print("-" * 50)
    correctness_scores = df_results['correctness_numeric']
    total_tests = len(df_results)
    correct_answers = correctness_scores.sum()
    accuracy = (correct_answers / total_tests) * 100

    print(f"Total Test Cases: {total_tests}")
    print(f"Correct Answers: {int(correct_answers)}")
    print(f"Incorrect Answers: {total_tests - int(correct_answers)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Execution Time: {df_results['execution_time'].mean():.2f} seconds")
    print(f"Total Execution Time: {df_results['execution_time'].sum():.2f} seconds")
    
    # Property-wise Performance
    print("\nPROPERTY-WISE PERFORMANCE")
    print("-" * 50)
    property_performance = df_results.groupby('inputs.property_id').agg({
        'correctness_numeric': ['count', 'sum', 'mean'],
        'execution_time': ['mean', 'std']
    }).round(2)

    property_performance.columns = ['Total_Tests', 'Correct_Answers', 'Accuracy', 'Avg_Time', 'Time_Std']
    property_performance['Accuracy'] = (property_performance['Accuracy'] * 100).round(2)
    property_performance = property_performance.sort_values('Accuracy', ascending=False)

    print(property_performance)

    failed_tests = df_results[df_results['correctness_numeric'] == 0]
    
    # Performance Insights
    print(f"\nPerformance Insights:")
    print("-" * 30)
    fastest_test = df_results.loc[df_results['execution_time'].idxmin()]
    slowest_test = df_results.loc[df_results['execution_time'].idxmax()]
    print(f"Fastest Response: {fastest_test['execution_time']:.2f}s - Property {fastest_test['inputs.property_id']}")
    print(f"Slowest Response: {slowest_test['execution_time']:.2f}s - Property {slowest_test['inputs.property_id']}")
    print(f"Most Accurate Property: {property_performance.index[0]} ({property_performance.iloc[0]['Accuracy']:.1f}%)")
    print(f"Least Accurate Property: {property_performance.index[-1]} ({property_performance.iloc[-1]['Accuracy']:.1f}%)")
    
    # Statistical Summary
    print(f"\nSTATISTICAL SUMMARY")
    print("-" * 30)
    print(f"Confidence Interval (95%): {accuracy:.2f}% ± {1.96 * np.sqrt(accuracy * (100 - accuracy) / total_tests):.2f}%")
    print(f"Standard Error: {np.sqrt(accuracy * (100 - accuracy) / total_tests):.2f}%")

    if total_tests > 1:
        print(f"Performance Consistency: {100 - (correctness_scores.std() * 100):.2f}%")
    
    return df_results


def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Initialize LangSmith client
    client = Client()
    
    # Create dataset
    create_dataset(client, CSV_URL, DATASET_NAME)
    
    # Run evaluation
    if SHOULD_EVALUATE_DATASET:
        print("\nStarting evaluation...")
        result = client.evaluate(
            get_target_output,
            data=DATASET_NAME,
            evaluators=[correctness_evaluator, tokens, cost],
            experiment_prefix=EXPERIMENT_NAME,
        )
        
        print(f"\nEvaluation completed. Result: {result}")
        
        # Analyze results
        df_results = analyze_results(result)
        
        return result, df_results
    
    return None, None


if __name__ == "__main__":
    result, df_results = main()