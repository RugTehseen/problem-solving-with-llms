#!/usr/bin/env python3
"""
Bi-Semantic Entropy Calculator

This script calculates the bi-semantic entropy for a given input question using
Ollama with the gemma3:4b model. It generates semantically equivalent formulations
of the question, defines semantic answer categories, and computes the entropy.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ollama
import time
import math
import os
from typing import List, Dict, Tuple, Any

def call_ollama_chat(prompt: str, sys_prompt: str = None, model: str = "gemma3:4b", temperature: float = 1.3) -> Dict[str, Any]:
    """
    Call ollama.chat with the specified prompt, model, and temperature.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: gemma3:4b)
        temperature: The temperature parameter (default: 1.3)
        
    Returns:
        The response from ollama.chat
    """
    messages = [{"role": "user", "content": prompt}]
    if sys_prompt:
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}] 

    return ollama.chat(
        model=model,
        messages=messages,
        options={
            'temperature': temperature
        }
    )

# Constants
NUM_FORMULATIONS = 4
NUM_CATEGORIES = 4
NUM_ANSWERS_PER_FORMULATION = 4  # Number of answers to generate per formulation

def generate_equivalent_formulations(question: str, num_formulations: int = NUM_FORMULATIONS) -> List[str]:
    """
    Generate semantically equivalent formulations of the input question.
    
    Args:
        question: The original question
        num_formulations: Number of equivalent formulations to generate
        
    Returns:
        List of semantically equivalent formulations
    """
    print(f"\n🔄 Generating {num_formulations} semantically equivalent formulations for: '{question}'")
    
    formulations = [question]  # Include the original question
    
    for i in range(1, num_formulations):
        prompt = f"""
        Create a creative but semantically equivalent formulation of the following question.
        The new formulation should ask for the same information content but use different wordings.
        Do not use the formulations that you already created.
        
        Original question: {question}

        Formulations already created: "\n".join({formulations})
        
        Return ONLY the reformulated question, nothing else.
        """
        
        print(f"  Generating formulation {i+1}/{num_formulations}...")
        
        try:
            response = call_ollama_chat(prompt)
            
            formulation = response['message']['content'].strip()
            formulations.append(formulation)
            print(f"  ✓ Formulation {i+1}: '{formulation}'")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ Error generating formulation {i+1}: {e}")
            formulations.append(f"{question} (variation {i+1})")
    
    return formulations

def generate_answer_categories(question: str, num_categories: int = NUM_CATEGORIES) -> List[str]:
    """
    Generate distinct semantic answer categories for the input question.
    
    Args:
        question: The original question
        num_categories: Number of semantic categories to generate
        
    Returns:
        List of semantic answer categories
    """
    print(f"\n🏷️ Generating {num_categories} semantic answer categories for: '{question}'")
    
    prompt = f"""
    For the following question, identify {num_categories} distinct semantic categories that answers might fall into.
    You can be creative in the choice of categories. Even for questions
    where there is one obvious category come up with other categories as well.
    
    Question: {question}
    
    Return your response as a JSON array of strings, where each string is a self-explanatory but short category name such as "Finance" or "2nd Grade Math".
    Example format: ["Category 1", "Category 2", "Category 3"]
    """
    
    try:
        response = call_ollama_chat(prompt)
        
        content = response['message']['content'].strip()
        
        # Extract JSON array from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            categories = json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            categories = [line.replace('- ', '').replace('"', '').replace("'", "") for line in lines[:num_categories]]
        
        # Ensure we have exactly num_categories
        if len(categories) < num_categories:
            categories.extend([f"Category {i+1}" for i in range(len(categories), num_categories)])
        elif len(categories) > num_categories:
            categories = categories[:num_categories]
        
        for i, category in enumerate(categories):
            print(f"  ✓ Category {i+1}: '{category}'")
        
        # Let us add a "Unrelated" Category for all answers that do not fall in either of the generated categories
        categories.append("Unrelated")
        return categories
        
    except Exception as e:
        print(f"  ✗ Error generating answer categories: {e}")
        # Fallback categories
        categories = [f"Category {i+1}" for i in range(num_categories)]
        for i, category in enumerate(categories):
            print(f"  ✓ Default Category {i+1}: '{category}'")
        return categories

def generate_answers(formulation: str, num_answers: int = NUM_ANSWERS_PER_FORMULATION, sys_prompt: str = None) -> List[str]:
    """
    Generate answers for a given question formulation.
    
    Args:
        formulation: The question formulation
        num_answers: Number of answers to generate
        sys_prompt: Optional system prompt to control the AI's response style
        
    Returns:
        List of generated answers
    """
    print(f"\n📝 Generating {num_answers} answers for formulation: '{formulation}'")
    
    answer = None
    answers = []
    
    # Use provided system prompt or default to the blog writer prompt
    if sys_prompt is None:
        sys_prompt = ""
    for i in range(num_answers):
        try:
            if answer is not None:
                sys_prompt + "\n-" + f"{answer}"
                if len(answer) > 50:
                    sys_prompt = sys_prompt + "\n-" + f"{answer[:50]}"
            response = call_ollama_chat(formulation, sys_prompt=sys_prompt, temperature=0)
            
            answer = response['message']['content'].strip()
            answers.append(answer)
            print(f"  ✓ Answer {i+1}: '{answer[:250]}' ({len(answer)} chars)")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ Error generating answer {i+1}: {e}")
            answers.append(f"Failed to generate answer {i+1}")
    
    return answers

def classify_answer(answer: str, categories: List[str], question: str) -> int:
    """
    Classify an answer into one of the semantic categories.
    
    Args:
        answer: The answer to classify
        categories: List of semantic categories
        question: The original question
        
    Returns:
        Index of the category
    """
    category_descriptions = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
    
    prompt = f"""
    Classify the following answer into one of the semantic categories below.
    
    Question: {question}
    Answer: {answer}
    
    Categories:
    {category_descriptions}
    
    Return ONLY the category number (1 to {len(categories)}) that best matches the answer.
    """
    
    try:
        response = call_ollama_chat(prompt, temperature=0)
        
        content = response['message']['content'].strip()
        
        # Extract the category number
        for char in content:
            if char.isdigit() and 1 <= int(char) <= len(categories):
                return int(char) - 1
        
        # If no valid digit found, return a random category
        return np.random.randint(0, len(categories))
        
    except Exception as e:
        print(f"  ✗ Error classifying answer: {e}")
        # Return a random category on error
        return np.random.randint(0, len(categories))

def calculate_bi_semantic_entropy(classification_counts: np.ndarray) -> float:
    """
    Calculate the bi-semantic entropy from classification counts, normalized to [0,1].
    
    Args:
        classification_counts: 2D array where rows are formulations and columns are categories
        
    Returns:
        Normalized bi-semantic entropy value (between 0 and 1)
    """
    K, M = classification_counts.shape  # K formulations, M categories
    N = np.sum(classification_counts[0])  # Answers per formulation
    
    # Calculate global counts for each class
    global_counts = np.sum(classification_counts, axis=0)
    
    # Calculate global probabilities
    global_probs = global_counts / (K * N)
    
    # Calculate Shannon entropy
    entropy = 0
    for p in global_probs:
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log2(p)
    
    # Calculate maximum possible entropy for normalization
    max_entropy = math.log2(M)
    
    # Normalize entropy to range [0,1]
    if max_entropy > 0:  # Avoid division by zero
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0
    
    return normalized_entropy

def visualize_results(classification_counts: np.ndarray, categories: List[str], formulations: List[str], entropy: float):
    """
    Visualize the classification results and entropy.
    
    Args:
        classification_counts: 2D array where rows are formulations and columns are categories
        categories: List of semantic categories
        formulations: List of question formulations
        entropy: Calculated bi-semantic entropy
    """
    # Create a figure with two subplots with increased size and better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Stacked bar chart of classification counts
    df = pd.DataFrame(classification_counts, columns=categories)
    df.index = [f"F{i+1}" for i in range(len(formulations))]
    
    df.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Answer Classifications by Formulation')
    ax1.set_xlabel('Question Formulation')
    ax1.set_ylabel('Number of Answers')
    
    # Plot 2: Pie chart of global distribution
    global_counts = np.sum(classification_counts, axis=0)
    
    # Adjust pie chart to handle long category names better
    wedges, texts, autotexts = ax2.pie(
        global_counts,
        labels=None,  # We'll add labels separately
        autopct='%1.1f%%',
        textprops={'fontsize': 9}
    )
    
    # Add a legend for the pie chart instead of direct labels
    ax2.legend(
        wedges,
        categories,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    ax2.set_title(f'Global Distribution of Semantic Categories\nBi-Semantic Entropy: {entropy:.4f}')
    
    # Use tight_layout with adjusted parameters
    plt.subplots_adjust(wspace=0.3, right=0.85)
    
    # Save and show the figure
    plt.savefig('bi_semantic_entropy_results.png', bbox_inches='tight')
    print("\n📊 Visualization saved as 'bi_semantic_entropy_results.png'")
    plt.show()

def save_entropy_to_dataframe(question: str, sys_prompt: str, run_id: int, formulations: List[str],
                     categories: List[str], classification_counts: np.ndarray, entropy: float,
                     csv_path: str = 'bi_semantic_entropy_data.csv'):
    """
    Save the bi-semantic entropy data to a pandas DataFrame and export as CSV.
    
    Args:
        question: The original question
        sys_prompt: The system prompt used
        run_id: The run identifier
        formulations: List of question formulations
        categories: List of semantic categories
        classification_counts: 2D array of classification counts
        entropy: The calculated bi-semantic entropy
        csv_path: Path to save the CSV file
    """
    # Create a new row of data
    data = {
        'question': question,
        'sys_prompt': sys_prompt,
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'formulations': len(formulations),
        'answers_per_formulation': classification_counts.sum() // len(formulations),
        'entropy': entropy
    }
    
    # Add category names and counts
    global_counts = np.sum(classification_counts, axis=0)
    for i, category in enumerate(categories):
        data[f'category_{i+1}'] = category
        data[f'category_{i+1}_count'] = global_counts[i]
    
    # Create a DataFrame with the new row
    new_row_df = pd.DataFrame([data])
    
    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        # Load existing data and append the new row
        try:
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            print(f"\n📊 Updating existing CSV file with new data row at '{csv_path}'")
        except Exception as e:
            print(f"\n⚠️ Error reading existing CSV: {e}. Creating a new file.")
            updated_df = new_row_df
    else:
        # Create a new CSV file
        updated_df = new_row_df
        print(f"\n📊 Creating new CSV file at '{csv_path}'")
    
    # Save the DataFrame to CSV
    updated_df.to_csv(csv_path, index=False)
    print(f"✅ Data saved successfully to '{csv_path}'")
    
    return updated_df