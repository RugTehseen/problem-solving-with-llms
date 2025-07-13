#!/usr/bin/env python3
"""
Emotional Valence Analyzer

This script:
1. Takes a question from the user
2. Gets an answer from gemma3:4b via Ollama
3. Has 10 different personas rate their happiness with the answer (1-10 scale)
4. Calculates emotional valence (mean happiness) and variance
5. Visualizes the results
"""

import ollama
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from typing import List, Dict, Any

# Define the personas with their prompts
PERSONAS = {
    "Analytical Alex": "You are a mathematician and data scientist who finds beauty in precise information. Your satisfaction comes from logically sound arguments backed by quantifiable evidence. You become frustrated with vague generalizations and emotional appeals. Clear definitions, specific numbers, and methodical explanations are what truly resonate with you.",

    "Optimistic Olivia": "As a motivational speaker and life coach, you believe in the transformative power of positive framing. You appreciate responses that highlight possibilities rather than limitations, and that offer hope while still being grounded in reality. Solutions-oriented perspectives that empower others delight you, while dwelling on problems without pathways forward feels deeply unsatisfying.",

    "Skeptical Sam": "With your background in investigative journalism, you've developed a natural distrust of easy answers. You value critical thinking that exposes underlying assumptions and considers counterarguments. Simple, one-sided explanations leave you cold, but you experience genuine intellectual pleasure when someone presents nuanced perspectives that acknowledge complexity.",

    "Pragmatic Pat": "Your experience running small businesses has made you value efficiency and real-world application above all. Abstract theories without clear implementation steps seem pointless to you. You appreciate concise, actionable insights that can be immediately applied to produce tangible results.",

    "Methodical Morgan": "As a systems engineer, you find satisfaction in well-structured, comprehensive approaches. You appreciate responses that build logically from fundamentals to advanced concepts with clear organization. Scattered thoughts and improper sequencing of ideas frustrate you, while well-categorized information with proper hierarchies brings you professional joy.",

    "Empathetic Emma": "Your background in counseling psychology has attuned you to the human element in every situation. You value responses that consider emotional impact and ethical implications. Technical solutions that ignore human needs feel hollow to you, while insights that balance practical concerns with compassion for affected individuals deeply resonate.",

    "Innovative Ian": "As a design thinking consultant, you're constantly seeking creative breakthroughs. Conventional wisdom and standard approaches bore you immensely. You're energized by unexpected connections between ideas, novel frameworks that challenge assumptions, and imaginative solutions that others haven't considered.",

    "Detailed Dana": "With your background in quality assurance, you believe the devil is truly in the details. You value comprehensive responses that leave no stone unturned. Surface-level explanations that gloss over important nuances frustrate you greatly, while thorough analyses covering edge cases and specific examples bring you immense satisfaction.",

    "Cautious Chris": "Your experience in risk management has taught you to value careful consideration of potential downsides. You appreciate balanced perspectives that acknowledge limitations and potential issues. Overly confident claims without proper qualification concern you, while measured responses that thoughtfully weigh pros and cons feel trustworthy and satisfying.",

    "Dynamic Devon": "As an emergency response coordinator, you thrive in rapidly changing environments. You value adaptive thinking and flexible approaches rather than rigid frameworks. You appreciate responses that offer multiple strategies that can be adjusted according to circumstances. Static, one-size-fits-all solutions feel constraining to you."
}

def call_ollama_chat(prompt: str, sys_prompt: str = None, model: str = "gemma3:4b", temperature: float = 0.7) -> Dict[str, Any]:
    """
    Call ollama.chat with the specified prompt, model, and temperature.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use (default: gemma3:4b)
        temperature: The temperature parameter (default: 0.7)
        
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

def get_answer(question: str, sys_prompt: str = None) -> str:
    """
    Get an answer to a question from the LLM.
    
    Args:
        question: The question
        sys_prompt: Optional system prompt to control the AI's response style
        
    Returns:
        The model's answer
    """
    print(f"\n🤔 Getting answer for question: '{question}'")
    
    # Use provided system prompt or default to professional one
    if not sys_prompt:
        sys_prompt = ""
    try:
        response = call_ollama_chat(question, sys_prompt=sys_prompt, temperature=0)
        answer = response['message']['content'].strip()
        print(f"\n✅ Received answer ({len(answer)} chars)")
        print(f"Answer preview: '{answer[:150]}...'")
        return answer
    except Exception as e:
        print(f"\n❌ Error getting answer: {e}")
        return f"Error: Failed to get an answer for the question: {question}"

def get_persona_rating(persona_name: str, persona_prompt: str, question: str, answer: str) -> int:
    """
    Get a happiness rating from a persona for the given answer.
    
    Args:
        persona_name: Name of the persona
        persona_prompt: The persona's prompt/description
        question: The original question
        answer: The answer to rate
        
    Returns:
        Happiness rating (1-10)
    """
    print(f"\n👤 Getting rating from {persona_name}...")
    
    prompt = f"""
    {persona_prompt}
    
    You are evaluating a response to the following question:
    
    Question: {question}
    
    Response: {answer}
    
    On a scale from 1 (not at all happy) to 10 (totally excited), rate your happiness with this response.
    
    Return ONLY a single number between 1 and 10 representing your happiness rating. No explanation or other text.
    """
    
    try:
        response = call_ollama_chat(prompt, temperature=0)
        content = response['message']['content'].strip()
        
        # Extract the rating number
        for word in content.split():
            # Remove any non-digit characters
            clean_word = ''.join(c for c in word if c.isdigit())
            if clean_word and 1 <= int(clean_word) <= 10:
                rating = int(clean_word)
                print(f"  ✓ {persona_name}'s rating: {rating}/10")
                return rating
        
        # If no valid rating found, return a default
        print(f"  ⚠️ Could not extract rating from: '{content}'. Using default.")
        return 5
    except Exception as e:
        print(f"  ❌ Error getting rating: {e}")
        return 5  # Default rating on error

def calculate_emotional_metrics(ratings: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate emotional valence (mean) and variance from ratings.
    
    Args:
        ratings: Dictionary mapping persona names to their ratings
        
    Returns:
        Dictionary with emotional metrics
    """
    values = list(ratings.values())
    
    # Calculate metrics
    mean = np.mean(values)
    variance = np.var(values)
    std_dev = np.std(values)
    
    return {
        "valence": mean,
        "variance": variance,
        "std_dev": std_dev
    }

def truncate_strings(string_list):
    result = []
    for s in string_list:
        if len(s) <= 7:
            result.append(s)
        else:
            result.append(s[:7] + "...")
    return result

def visualize_results(ratings: Dict[str, int], metrics: Dict[str, float], question: str):
    """
    Visualize the emotional valence results.
    
    Args:
        ratings: Dictionary mapping persona names to their ratings
        metrics: Dictionary with emotional metrics
        question: The original question
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sort personas by rating for better visualization
    sorted_items = sorted(ratings.items(), key=lambda x: x[1])
    personas = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    xticks = truncate_strings(personas)
    
    # Plot 1: Bar chart of individual ratings
    colors = plt.cm.RdYlGn(np.array(values) / 10.0)  # Red to yellow to green color map
    ax1.bar(xticks, values, color=colors)
    ax1.set_title('Happiness Ratings by Persona')
    ax1.set_xlabel('Persona')
    ax1.set_ylabel('Happiness Rating (1-10)')
    ax1.set_ylim(0, 11)  # Set y-axis limit to 0-11 for better visualization
    ax1.tick_params(axis='x', rotation=45)
    
    # Add the actual values on top of each bar
    for i, v in enumerate(values):
        ax1.text(i, v + 0.3, str(v), ha='center')
    
    # Plot 2: Gauge chart for emotional valence
    valence = metrics['valence']
    
    # Create a half-circle gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    
    # Calculate the position for the needle
    needle_theta = np.pi * (1 - (valence - 1) / 9)  # Map 1-10 to pi-0
    needle_x = r * np.cos(needle_theta)
    needle_y = r * np.sin(needle_theta)
    
    # Plot the gauge background
    ax2.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)
    
    # Add colored arcs for different ranges
    theta_ranges = [
        (0, np.pi/3),        # Red (1-3.67)
        (np.pi/3, 2*np.pi/3), # Yellow (3.67-6.33)
        (2*np.pi/3, np.pi)    # Green (6.33-10)
    ]
    colors = ['green', 'gold', 'red']
    
    for (theta_min, theta_max), color in zip(theta_ranges, colors):
        theta_range = np.linspace(theta_min, theta_max, 30)
        ax2.plot(r * np.cos(theta_range), r * np.sin(theta_range), '-', color=color, linewidth=10, alpha=0.7)
    
    # Plot the needle
    ax2.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
    ax2.plot([0], [0], 'ko', markersize=10)
    
    # Add labels
    ax2.text(-0.9, -0.15, '1', fontsize=12)
    ax2.text(0, -0.15, '5.5', fontsize=12)
    ax2.text(0.9, -0.15, '10', fontsize=12)
    
    # Add the valence value
    ax2.text(0, -0.5, f'Emotional Valence: {valence:.2f}', ha='center', fontsize=14)
    ax2.text(0, -0.65, f'Variance: {metrics["variance"]:.2f}', ha='center', fontsize=14)
    
    # Remove axes and set equal aspect ratio
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    # Set title for the entire figure
    plt.suptitle(f'Emotional Response Analysis for Question:\n"{question}"', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save and show the figure
    plt.savefig('emotional_valence_results.png', bbox_inches='tight')
    print("\n📊 Visualization saved as 'emotional_valence_results.png'")
    plt.show()

def save_emotional_valence_to_dataframe(sys_prompt: str, ratings: Dict[str, int], metrics: Dict[str, float], csv_path: str = 'emotional_valence_data.csv'):
    """
    Save the emotional valence data to a pandas DataFrame and export as CSV.
    
    Args:
        sys_prompt: The system prompt used for the analysis
        ratings: Dictionary mapping persona names to their ratings
        metrics: Dictionary with emotional metrics
        csv_path: Path to save the CSV file
    """
    # Create a new row of data
    data = {'sys_prompt': sys_prompt}
    
    # Add persona ratings
    for persona, rating in ratings.items():
        data[f"{persona}_score"] = rating
    
    # Add metrics
    data['mean_valence'] = metrics['valence']
    data['variance'] = metrics['variance']
    
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