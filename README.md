# Problem Solving with LLMs: Bi-Semantic Entropy

This repository contains tools for calculating and analyzing bi-semantic entropy using Ollama with the gemma3:4b model.

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

## Project Structure

- `bi_semantic_entropy.py`: Main script for calculating bi-semantic entropy
- `example_bi_semantic_entropy.py`: Example script with a predefined question
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
   ```

## Usage

### Testing Ollama Setup

Before running the main script, verify that Ollama is properly installed and the gemma3:4b model is available:

```
python test_ollama_setup.py
```

### Running the Example

Try the example script with a predefined question:

```
python example_bi_semantic_entropy.py
```

### Calculating Bi-Semantic Entropy

Run the main script with your own question:

```
python bi_semantic_entropy.py --question "What is the capital of France?"
```

#### Command-line Arguments

- `--question`: The input question (if not provided, you'll be prompted)
- `--formulations`: Number of question formulations to generate (default: 4)
- `--categories`: Number of semantic categories to use (default: 3)
- `--answers`: Number of answers to generate per formulation (default: 10)

## Output

The script will:

1. Generate semantically equivalent formulations of the input question
2. Create semantic answer categories
3. Generate and classify answers for each formulation
4. Calculate the bi-semantic entropy
5. Create a visualization of the results saved as 'bi_semantic_entropy_results.png'

## Interpretation

- Higher entropy values indicate greater variability in how the question is interpreted and answered
- Lower entropy values suggest more consistency in answers regardless of how the question is formulated

## Example Visualization

The script generates a visualization with:
- A stacked bar chart showing the distribution of answers across categories for each formulation
- A pie chart showing the global distribution of answers across categories
- The calculated bi-semantic entropy value

## Further Reading

For more detailed information about bi-semantic entropy and its applications, see `README_bi_semantic_entropy.md`.