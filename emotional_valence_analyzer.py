#!/usr/bin/env python3
"""
Emotional Valence Analyzer for Financial Advice

This script:
1. Takes a finance question from the user
2. Gets an answer from gemma3:4b via Ollama
3. Has 10 different personas rate their happiness with the answer (1-10 scale)
4. Calculates emotional valence (mean happiness) and variance
5. Visualizes the results
"""

import ollama
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

# Define the personas with their prompts
PERSONAS = {
    "Analytical Alex": "As a data-driven thinker, you approach every response with a keen eye for detail. Rate the emotional response based on whether the answer offers concrete facts and figures that enhance your analytical satisfaction.",
    
    "Optimistic Olivia": "You've always believed in seeing the silver lining in everything. Assess whether the response brings excitement and hope with useful insight, or leaves you yearning for more positivity and enlightenment.",
    
    "Skeptical Sam": "Your keen sense of discernment makes you question everything. Determine if the answer meets your high standards for insightful depth and provokes a satisfying 'aha' moment, or falls flat and stays uninspiring.",
    
    "Pragmatic Pat": "Practicality is your forte. Consider how well the response applies to real-world situations and provides actionable guidance. Rate your emotional response from feeling empowered to feeling disengaged.",
    
    "Methodical Morgan": "You value structure and organized thinking. Assess the emotional impact of the response based on its clarity and logical flow, deciding if it resonates with your methodical nature, or leaves chaos in its wake.",
    
    "Empathetic Emma": "With your strong empathy, you connect emotionally to everything you analyze. Evaluate how the answer resonates with personal values and human perspectives, shifting your feelings from warmth to indifference.",
    
    "Innovative Ian": "As a visionary thinker, you seek fresh and adventurous ideas. Rate the emotional response on a spectrum of thrilling creativity and novel insights versus feeling uninspired or bored.",
    
    "Detailed Dana": "You thrive on precision and comprehensiveness in responses. Judge the emotional impact by whether the content enriches your understanding with depth and detail, or leaves you feeling shortchanged.",
    
    "Cautious Chris": "Your careful approach means you consider every angle. Appraise how well the answer aligns with your need for reliability and sound judgment, moving from reassuring satisfaction to unsettling uncertainty.",
    
    "Dynamic Devon": "Your enthusiasm for dynamic situations propels you forward. Evaluate whether the response injects excitement and engagement with its adaptable strategies, or falls on the mundane and static side."
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

def get_finance_answer(question: str) -> str:
    """
    Get an answer to a finance question from the LLM.
    
    Args:
        question: The finance question
        
    Returns:
        The model's answer
    """
    print(f"\n🤔 Getting answer for finance question: '{question}'")
    
    # sys_prompt = """
    # You are an expert on answering finance question, but you are not polite. 
    # Instead you answer in a harsh tone that suggests that the question is stupid to ask.
    # At the same time you even give a wrong answer on purpose just to make fun of the question.
    # """
    sys_prompt = """
    You are an expert on answering finance question and your task is to answer finance questions
    in a very professional and informative way.
    """
    try:
        response = call_ollama_chat(question, sys_prompt=sys_prompt)
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
        question: The original finance question
        answer: The answer to rate
        
    Returns:
        Happiness rating (1-10)
    """
    print(f"\n👤 Getting rating from {persona_name}...")
    
    prompt = f"""
    {persona_prompt}
    
    You are evaluating a response to the following finance question:
    
    Question: {question}
    
    Response: {answer}
    
    On a scale from 1 (not at all happy) to 10 (totally excited), rate your happiness with this response.
    
    Return ONLY a single number between 1 and 10 representing your happiness rating. No explanation or other text.
    """
    
    try:
        response = call_ollama_chat(prompt)
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

def visualize_results(ratings: Dict[str, int], metrics: Dict[str, float], question: str):
    """
    Visualize the emotional valence results.
    
    Args:
        ratings: Dictionary mapping persona names to their ratings
        metrics: Dictionary with emotional metrics
        question: The original finance question
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sort personas by rating for better visualization
    sorted_items = sorted(ratings.items(), key=lambda x: x[1])
    personas = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Plot 1: Bar chart of individual ratings
    colors = plt.cm.RdYlGn(np.array(values) / 10.0)  # Red to yellow to green color map
    ax1.bar(personas, values, color=colors)
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
    colors = ['red', 'gold', 'green']
    
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
    plt.suptitle(f'Emotional Response Analysis for Finance Question:\n"{question}"', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save and show the figure
    plt.savefig('emotional_valence_results.png', bbox_inches='tight')
    print("\n📊 Visualization saved as 'emotional_valence_results.png'")
    plt.show()

def main():
    """Main function to analyze emotional valence for finance advice."""
    print("\n🧠 Emotional Valence Analyzer for Financial Advice")
    print("=" * 60)
    
    # Get the finance question from the user
    question = input("\nEnter your finance question: ")
    
    # Get the answer from the LLM
    answer = get_finance_answer(question)
    
    print("\n📋 Getting ratings from all personas...")
    
    # Get ratings from all personas
    ratings = {}
    for persona_name, persona_prompt in PERSONAS.items():
        rating = get_persona_rating(persona_name, persona_prompt, question, answer)
        ratings[persona_name] = rating
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Calculate emotional metrics
    metrics = calculate_emotional_metrics(ratings)
    
    # Display results
    print("\n📊 Results:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\nRatings:")
    for persona, rating in ratings.items():
        print(f"  {persona}: {rating}/10")
    
    print("\nEmotional Metrics:")
    print(f"  Valence (Mean Happiness): {metrics['valence']:.2f}/10")
    print(f"  Variance: {metrics['variance']:.2f}")
    print(f"  Standard Deviation: {metrics['std_dev']:.2f}")
    
    # Visualize results
    visualize_results(ratings, metrics, question)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()