# Solution Quality Analysis App: Bi-Semantic Entropy and Emotional Valence

This repository contains a Streamlit application for calculating and analyzing bi-semantic entropy and emotional valence using Ollama with the gemma3:4b model. The app provides an interactive interface to explore how different formulations of questions affect answers and how different personas respond emotionally to AI-generated content.

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

## Streamlit Application

The repository includes a Streamlit web application (`app.py`) that provides a user-friendly interface for both analysis tools:

### Emotional Valence Tool
- Input a use case and generate a diverse group of users
- Input a question and get an AI-generated answer
- Customize the system prompt to control how the AI responds (professional, casual, creative, etc.)
- See how the generated personas with varying perspectives rate the answer
- Visualize the emotional valence (average rating) and variance across personas

### Bi-Semantic Entropy Tool
- Input a question and analyze how different formulations impact normalized bi-semantic entropy
- Customize the number of runs, as well as formulations per run and generated answers per formulation
- Specify your own custom categories an answer might fall into
- Visualize the distribution of answers across categories and calculate entropy

## Project Structure

- `app.py`: Streamlit web application providing a GUI for both analysis tools
- `bi_semantic_entropy.py`: Core implementation of the bi-semantic entropy calculator
- `emotional_valence_analyzer.py`: Core implementation of the emotional valence analyzer
- `requirements.txt`: List of required Python packages
- `evaluations`: Contains evaluation results for dedicated scenarios that were analyzed

## Prerequisites

- Python 3.6+
- Ollama installed with the gemma3:4b model
- Required Python packages: ollama, pandas, numpy, matplotlib, streamlit

## Installation

1. Ensure you have Ollama installed:
   ```
   # Follow instructions at https://ollama.com/download
   ```

2. Pull the gemma3:4b model:
   ```
   ollama pull gemma3:4b
   ```

3. Create and activate virtual environment
   ```
   python -m venv <ENVIRONMENT_NAME>
   ```
   ```
   <ENVIRONMENT_NAME>\Scripts\activate (Windows) 
   ``` 
   ```
   source <ENVIRONMENT_NAME>/bin/activate (Unix)
   ```

4. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Run the Streamlit application with:
```
streamlit run app.py
```

This will start the web server and open the application in your default web browser.