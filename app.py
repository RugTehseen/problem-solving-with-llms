import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import os
from PIL import Image
import time
import threading
import queue

# Import functions from the bi_semantic_entropy script
from bi_semantic_entropy import (
    generate_equivalent_formulations,
    generate_answer_categories,
    generate_answers,
    classify_answer,
    calculate_bi_semantic_entropy,
    visualize_results
)

# Import functions from the emotional_valence_analyzer script
from emotional_valence_analyzer import (
    get_answer,
    get_persona_rating,
    calculate_emotional_metrics,
    visualize_results as visualize_valence_results,
    PERSONAS
)

# Set page configuration
st.set_page_config(
    page_title="Solution Quality Analysis App",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for logs if it doesn't exist
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def add_log(message):
    st.session_state.log_messages.append(message)

# Main title and description
st.title("🧠 Solution Quality Analysis App")
st.markdown("""
This app provides two different NLP analysis tools:
- **Bi-Semantic Entropy**: Analyzes the semantic diversity of answers to a question
- **Emotional Valence**: Analyzes the emotional response to an answer
""")

# Create a sidebar for tool selection
analysis_type = st.sidebar.radio(
    "Select Analysis Tool",
    ["Bi-Semantic Entropy", "Emotional Valence"]
)

# Custom stdout class to capture and redirect print statements
class LoggerStdout:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = ""
    
    def write(self, message):
        self.terminal.write(message)
        self.buffer += message
        if '\n' in message:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:  # Process all complete lines
                if line.strip():  # Only add non-empty lines
                    st.session_state.log_messages.append(line)
            self.buffer = lines[-1]  # Keep the last incomplete line in buffer
            
    def flush(self):
        self.terminal.flush()

# Function to run bi-semantic entropy analysis
def run_bi_semantic_entropy(question, num_formulations=4, num_answers=4, custom_categories=None):
    # Redirect standard output to capture print statements
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    
    logger_stdout = LoggerStdout()
    sys.stdout = logger_stdout
    
    try:
        # Step 1: Generate semantically equivalent formulations
        formulations = generate_equivalent_formulations(question, num_formulations)
        
        # Step 2: Use custom categories or generate semantic answer categories
        if custom_categories:
            # Use custom categories provided by the user
            categories = custom_categories
            
            print(f"\n🏷️ Using {len(categories)} custom categories:")
            for i, category in enumerate(categories):  # Skip the "Unrelated" category in the log
                print(f"  ✓ Category {i+1}: '{category}'")
        else:
            # Generate semantic answer categories
            categories = generate_answer_categories(question, num_categories=5)
        
        # Step 3: Generate answers for each formulation and classify them
        classification_counts = np.zeros((len(formulations), len(categories)), dtype=int)
        
        for i, formulation in enumerate(formulations):
            # Generate answers
            answers = generate_answers(formulation, num_answers)
            
            # Classify answers
            for j, answer in enumerate(answers):
                category_idx = classify_answer(answer, categories, question)
                classification_counts[i, category_idx] += 1
        
        # Step 4: Calculate bi-semantic entropy
        entropy = calculate_bi_semantic_entropy(classification_counts)
        
        # Step 5: Visualize results (save to file)
        plt.figure(figsize=(18, 8))
        visualize_results(classification_counts, categories, formulations, entropy)
        
        # Create a DataFrame for display
        df = pd.DataFrame(classification_counts,
                         index=[f"Formulation {i+1}" for i in range(len(formulations))],
                         columns=categories)
        
        # Restore standard output
        sys.stdout = old_stdout
        output_text = new_stdout.getvalue()
        
        return {
            "formulations": formulations,
            "categories": categories,
            "classification_counts": classification_counts,
            "entropy": entropy,
            "output_text": output_text,
            "dataframe": df
        }
    
    except Exception as e:
        sys.stdout = old_stdout
        return {"error": str(e)}

# Function to run emotional valence analysis
def run_emotional_valence(question, sys_prompt=None, custom_personas=None):
    # Redirect standard output to capture print statements
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    
    logger_stdout = LoggerStdout()
    sys.stdout = logger_stdout
    
    try:
        # Get the answer from the LLM
        answer = get_answer(question, sys_prompt)
        
        # Use custom personas if provided, otherwise use default PERSONAS
        personas_to_use = custom_personas if custom_personas else PERSONAS
        
        # Get ratings from all personas
        ratings = {}
        for persona_name, persona_prompt in personas_to_use.items():
            rating = get_persona_rating(persona_name, persona_prompt, question, answer)
            ratings[persona_name] = rating
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate emotional metrics
        metrics = calculate_emotional_metrics(ratings)
        
        # Visualize results (save to file)
        plt.figure(figsize=(18, 8))
        visualize_valence_results(ratings, metrics, question)
        
        # Restore standard output
        sys.stdout = old_stdout
        output_text = new_stdout.getvalue()
        
        return {
            "question": question,
            "answer": answer,
            "ratings": ratings,
            "metrics": metrics,
            "output_text": output_text,
            "custom_personas_used": custom_personas is not None,
            "personas_used": personas_to_use
        }
    
    except Exception as e:
        sys.stdout = old_stdout
        return {"error": str(e)}

# Main content based on selected tool
if analysis_type == "Bi-Semantic Entropy":
    st.header("🔄 Bi-Semantic Entropy Analysis")
    
    st.markdown("""
    This tool calculates the bi-semantic entropy for a given input question using
    Ollama with the gemma3:4b model. It generates semantically equivalent formulations
    of the question, defines semantic answer categories, and computes the entropy.
    """)
    
    # Input form
    with st.form("bi_semantic_form"):
        question = st.text_area("Enter your question:", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            num_formulations = st.slider("Number of formulations:", 2, 6, 4)
        with col2:
            num_answers = st.slider("Answers per formulation:", 2, 6, 4)
        
        custom_categories = None
        st.write("Enter your custom categories (one per line):")
        custom_categories_text = st.text_area("", height=150, placeholder=f"Define Categories")
        if custom_categories_text:
            custom_categories = [cat.strip() for cat in custom_categories_text.split('\n') if cat.strip()]
        
        submit_button = st.form_submit_button("Run Analysis")
    
    # Process when form is submitted
    if submit_button and question:
        with st.spinner("Running bi-semantic entropy analysis... This may take a few minutes."):
            # Clear previous log messages
            st.session_state.log_messages = []
            
            # Run analysis with log container
            results = run_bi_semantic_entropy(
                question,
                num_formulations,
                num_answers,
                custom_categories=custom_categories if custom_categories else None
            )
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                # Display results
                st.subheader("Results")
                
                # Display the entropy value
                st.metric("Bi-Semantic Entropy", f"{results['entropy']:.4f}")
                
                # Display the visualization
                if os.path.exists("bi_semantic_entropy_results.png"):
                    st.image("bi_semantic_entropy_results.png", use_container_width=True)
                
                # Display the classification counts
                st.subheader("Classification Counts")
                st.dataframe(results["dataframe"])
                
                # Display formulations
                st.subheader("Question Formulations")
                for i, form in enumerate(results["formulations"]):
                    st.write(f"**Formulation {i+1}:** {form}")
                
                # Display categories
                st.subheader("Semantic Categories")
                for i, cat in enumerate(results["categories"]):
                    st.write(f"**Category {i+1}:** {cat}")
                

elif analysis_type == "Emotional Valence":
    st.header("😊 Emotional Valence Analysis")
    
    st.markdown("""
    This tool analyzes the emotional valence of responses.
    It gets an answer from gemma3:4b via Ollama, has 10 different personas rate their
    happiness with the answer (1-10 scale), and calculates emotional valence (mean happiness) and variance.
    """)
    
    # Input form
    with st.form("emotional_valence_form"):
        question = st.text_area("Enter your question:", height=100)
        
        # Default professional system prompt
        default_sys_prompt = """You are an AI and your task is to answer questions
in a very professional and informative way."""
        
        # System prompt input
        st.write("Customize the AI's system prompt:")
        sys_prompt = st.text_area("System Prompt", value=default_sys_prompt, height=100,
                                help="This defines how the AI will respond to the question")
        
        # Personas customization
        st.write("Customize personas:")
        
        # Create a default personas string if not in session state
        if 'personas_text' not in st.session_state:
            personas_text = ""
            for name, prompt in PERSONAS.items():
                personas_text += f"{name}: {prompt}\n\n"
            st.session_state.personas_text = personas_text
        
        # Text area for personas customization
        personas_text = st.text_area(
            "Enter personas (format: 'Name: Description' with blank line between personas)",
            value=st.session_state.personas_text,
            height=300,
            help="Each persona should be in the format 'Name: Description' with a blank line between personas"
        )
        
        # Store the updated text in session state
        st.session_state.personas_text = personas_text
        
        # Parse the personas text into a dictionary
        custom_personas = {}
        current_name = None
        current_description = []
        
        for line in personas_text.split('\n'):
            line = line.strip()
            if ':' in line and current_name is None:
                # Start of a new persona
                parts = line.split(':', 1)
                current_name = parts[0].strip()
                current_description = [parts[1].strip()]
            elif line == '' and current_name is not None:
                # End of a persona description
                custom_personas[current_name] = '\n'.join(current_description)
                current_name = None
                current_description = []
            elif current_name is not None:
                # Continuation of a persona description
                current_description.append(line)
        
        # Add the last persona if there is one
        if current_name is not None:
            custom_personas[current_name] = '\n'.join(current_description)
        
        # Display warning if no personas are defined
        if len(custom_personas) == 0:
            st.warning("No personas defined. Please add at least one persona in the format 'Name: Description'.")
        
        submit_button = st.form_submit_button("Run Analysis")
    
    # Process when form is submitted
    if submit_button and question:
        with st.spinner("Running emotional valence analysis... This may take a few minutes."):
            # Clear previous log messages
            st.session_state.log_messages = []
            
            # Run analysis with log container
            run_analysis = True
            
            # Check if we have at least one persona
            if len(custom_personas) == 0:
                st.error("You must define at least one persona for the analysis.")
                run_analysis = False
            
            if run_analysis:
                results = run_emotional_valence(question, sys_prompt, custom_personas)
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                # Display results
                st.subheader("Results")
                
                # Display the metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Emotional Valence", f"{results['metrics']['valence']:.2f}/10")
                with col2:
                    st.metric("Variance", f"{results['metrics']['variance']:.2f}")
                with col3:
                    st.metric("Standard Deviation", f"{results['metrics']['std_dev']:.2f}")
                
                # Display the visualization
                if os.path.exists("emotional_valence_results.png"):
                    st.image("emotional_valence_results.png", use_container_width=True)
                
                # Display the answer
                st.subheader("AI Response to Question")
                st.write(results["answer"])
                
                # Display ratings
                st.subheader("Persona Ratings")
                ratings_df = pd.DataFrame(list(results["ratings"].items()), columns=["Persona", "Rating"])
                ratings_df = ratings_df.sort_values("Rating")
                st.dataframe(ratings_df)
                
                # Display which personas were used
                st.subheader("Personas Used")
                st.info(f"{len(results['personas_used'])} personas were used for this analysis.")
                
                # Show persona descriptions in expandable sections
                personas_used = results.get("personas_used", {})
                for persona_name, persona_desc in personas_used.items():
                    with st.expander(f"Persona: {persona_name}"):
                        st.write(persona_desc)
                