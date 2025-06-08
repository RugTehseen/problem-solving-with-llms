# Problem Solving with LLMs: Bi-Semantic Entropy and Emotional Valence

This repository contains tools for calculating and analyzing bi-semantic entropy and emotional valence using Ollama with the gemma3:4b model.

## What is Bi-Semantic Entropy?

Bi-Semantic Entropy (BSE) is a measure of how different but semantically equivalent formulations of a question lead to different answer categories. It quantifies the variability in semantic interpretations of a question.

### Formal Definition

Let Q = {q₁, q₂, …, q_K} be K distinct but semantically equivalent formulations of a question. For each formulation qₖ:

1. Generate a set of N distinct answers.
2. Classify each answer into one of M semantic categories.
3. Count how many answers from each formulation fall into each semantic category.
4. Calculate the global probability of an answer falling into each semantic category.
5. The bi-semantic entropy is the Shannon entropy over these probabilities:
   BSE(Q) = - ∑ⱼ₌₁ᴹ pⱼ log₂ pⱼ

## What is Emotional Valence?

Emotional Valence (VE) measures the emotional response elicited by an answer across different personas. It quantifies how different individuals with varying perspectives might emotionally react to the same information.

### Formal Definition

For an answer A to a question q, and a population of evaluators H, the Emotional Valence Function maps each evaluator's response to a scalar value:

VE(A,h) → R

The Expected Emotional Valence is the average across all evaluators:
E[VE(A)] = (1/|H|) ∑ VE(A,h)

The Emotional Variance measures how polarizing an answer is:
Var(VE(A)) = E[(VE(A) - E[VE(A)])²]

## Project Structure

- `bi_semantic_entropy.py`: Main script for calculating bi-semantic entropy
- `example_bi_semantic_entropy.py`: Example script with a predefined question
- `emotional_valence_analyzer.py`: Script for analyzing emotional valence of financial advice
- `test_ollama_setup.py`: Script to verify Ollama installation and model availability
- `README_bi_semantic_entropy.md`: Detailed documentation for the bi-semantic entropy calculator
- `requirements.txt`: List of required Python packages

## Prerequisites

- Python 3.6+
- Ollama installed with the gemma3:4b model
- Required Python packages: ollama, pandas, numpy, matplotlib

## Installation

1. Ensure you have Ollama installed:
   ```
   # Follow instructions at https://ollama.com/download
   ```

2. Pull the gemma3:4b model:
   ```
   ollama pull gemma3:4b
   ```

3. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Test your setup:
   ```
   python test_ollama_setup.py