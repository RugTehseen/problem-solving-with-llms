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
from typing import List, Dict, Tuple, Any

def call_ollama_chat(prompt: str, sys_prompt: str = None, model: str = "gemma3:4b", temperature: float = 1.3) -> Dict[str, Any]:
    """
    Call ollama.chat with the specified prompt, model, and temperature.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: gemma3:4b)
        temperature: The temperature parameter (default: 1)
        
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
NUM_CATEGORIES = 3
NUM_ANSWERS_PER_FORMULATION = 5  # Number of answers to generate per formulation

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
        
        Original question: {question}
        
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
    These categories should represent different approaches or perspectives to answering the question. Even for questions
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
        categories.append(["Unrelated"])
        return categories
        
    except Exception as e:
        print(f"  ✗ Error generating answer categories: {e}")
        # Fallback categories
        categories = [f"Category {i+1}" for i in range(num_categories)]
        for i, category in enumerate(categories):
            print(f"  ✓ Default Category {i+1}: '{category}'")
        return categories

def generate_answers(formulation: str, num_answers: int = NUM_ANSWERS_PER_FORMULATION) -> List[str]:
    """
    Generate answers for a given question formulation.
    
    Args:
        formulation: The question formulation
        num_answers: Number of answers to generate
        
    Returns:
        List of generated answers
    """
    print(f"\n📝 Generating {num_answers} answers for formulation: '{formulation}'")
    
    answers = []
    sys_prompt = """
    You are an AI dedicated to answering questions. However, you do not directly answer the question. Instead
    you are completely creative only scratching the true answer at the surface of you creative answer. 
    """
    for i in range(num_answers):
        try:
            response = call_ollama_chat(formulation, sys_prompt=sys_prompt)
            
            answer = response['message']['content'].strip()
            answers.append(answer)
            print(f"  ✓ Answer {i+1}: '{answer[:50]}...' ({len(answer)} chars)")
            
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
        response = call_ollama_chat(prompt)
        
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
    Calculate the bi-semantic entropy from classification counts.
    
    Args:
        classification_counts: 2D array where rows are formulations and columns are categories
        
    Returns:
        Bi-semantic entropy value
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
    
    return entropy

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

def main():
    """Main function to calculate bi-semantic entropy."""
    parser = argparse.ArgumentParser(description='Calculate bi-semantic entropy for a question.')
    parser.add_argument('--question', type=str, help='Input question')
    parser.add_argument('--formulations', type=int, default=NUM_FORMULATIONS, 
                        help=f'Number of question formulations (default: {NUM_FORMULATIONS})')
    parser.add_argument('--categories', type=int, default=NUM_CATEGORIES, 
                        help=f'Number of semantic categories (default: {NUM_CATEGORIES})')
    parser.add_argument('--answers', type=int, default=NUM_ANSWERS_PER_FORMULATION, 
                        help=f'Number of answers per formulation (default: {NUM_ANSWERS_PER_FORMULATION})')
    
    args = parser.parse_args()
    
    # Get the question from command line or prompt the user
    question = args.question
    if not question:
        question = input("Enter your question: ")
    
    print("\n🧠 Bi-Semantic Entropy Calculator")
    print("=" * 50)
    print(f"Question: {question}")
    print(f"Formulations: {args.formulations}")
    print(f"Categories: {args.categories}")
    print(f"Answers per formulation: {args.answers}")
    print("=" * 50)
    
    # Step 1: Generate semantically equivalent formulations
    formulations = generate_equivalent_formulations(question, args.formulations)
    
    # Step 2: Generate semantic answer categories
    categories = generate_answer_categories(question, args.categories)
    
    # Step 3: Generate answers for each formulation and classify them
    classification_counts = np.zeros((len(formulations), len(categories)), dtype=int)
    
    for i, formulation in enumerate(formulations):
        print(f"\n📋 Processing formulation {i+1}/{len(formulations)}: '{formulation}'")
        
        # Generate answers
        answers = generate_answers(formulation, args.answers)
        
        # Classify answers
        print(f"\n🔍 Classifying answers for formulation {i+1}...")
        for j, answer in enumerate(answers):
            category_idx = classify_answer(answer, categories, question)
            classification_counts[i, category_idx] += 1
            print(f"  ✓ Answer {j+1} classified as: '{categories[category_idx]}'")
    
    # Step 4: Calculate bi-semantic entropy
    entropy = calculate_bi_semantic_entropy(classification_counts)
    
    print("\n🔢 Classification Counts:")
    df = pd.DataFrame(classification_counts, 
                     index=[f"Formulation {i+1}" for i in range(len(formulations))],
                     columns=categories)
    print(df)
    
    print("######################## BI-ENTROPY #########################")
    print(f"\n🧮 Bi-Semantic Entropy: {entropy:.4f}")
    
    # Step 5: Visualize results
    visualize_results(classification_counts, categories, formulations, entropy)

if __name__ == "__main__":
    main()