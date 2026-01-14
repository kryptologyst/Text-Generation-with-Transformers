"""
Streamlit web interface for text generation with transformers.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_generator import TextGenerator, GenerationConfig, create_synthetic_dataset
from config import ConfigManager, create_default_config_file

# Page configuration
st.set_page_config(
    page_title="Text Generation with Transformers",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .generated-text {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'generated_texts' not in st.session_state:
        st.session_state.generated_texts = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}

@st.cache_resource
def load_text_generator(model_name: str, device: str, use_pipeline: bool):
    """Load and cache the text generator."""
    try:
        return TextGenerator(model_name=model_name, device=device, use_pipeline=use_pipeline)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🤖 Text Generation with Transformers</h1>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_options = {
        "GPT-2 (Small)": "gpt2",
        "GPT-2 Medium": "gpt2-medium",
        "GPT-2 Large": "gpt2-large",
        "DistilGPT-2": "distilgpt2"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    model_name = model_options[selected_model]
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        options=["auto", "cpu", "cuda"],
        index=0
    )
    
    # Generation parameters
    st.sidebar.header("🎛️ Generation Parameters")
    
    max_length = st.sidebar.slider("Max Length", 50, 500, 200)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.sidebar.slider("Top-k", 1, 100, 50)
    num_sequences = st.sidebar.slider("Number of Sequences", 1, 5, 1)
    
    # Load generator
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.generator = load_text_generator(model_name, device, True)
            if st.session_state.generator:
                st.sidebar.success("Model loaded successfully!")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate Text", "📊 Batch Processing", "📈 Evaluation", "ℹ️ About"])
    
    with tab1:
        st.header("Single Text Generation")
        
        if st.session_state.generator is None:
            st.warning("Please load a model first using the sidebar.")
        else:
            # Input prompt
            prompt = st.text_area(
                "Enter your prompt:",
                value="Once upon a time, in a land far, far away",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                generate_button = st.button("🚀 Generate", type="primary")
            
            if generate_button and prompt:
                with st.spinner("Generating text..."):
                    config = GenerationConfig(
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=num_sequences
                    )
                    
                    try:
                        if num_sequences == 1:
                            generated_text = st.session_state.generator.generate_text(prompt, config)
                            st.markdown('<div class="generated-text">', unsafe_allow_html=True)
                            st.write("**Generated Text:**")
                            st.write(generated_text)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            generated_texts = st.session_state.generator.generate_text(prompt, config)
                            for i, text in enumerate(generated_texts):
                                st.markdown('<div class="generated-text">', unsafe_allow_html=True)
                                st.write(f"**Generated Text {i+1}:**")
                                st.write(text)
                                st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating text: {e}")
    
    with tab2:
        st.header("Batch Text Generation")
        
        if st.session_state.generator is None:
            st.warning("Please load a model first using the sidebar.")
        else:
            # Batch input options
            input_method = st.radio(
                "Choose input method:",
                ["Manual Input", "Synthetic Dataset", "Upload File"]
            )
            
            prompts = []
            
            if input_method == "Manual Input":
                st.subheader("Enter multiple prompts:")
                prompt_text = st.text_area(
                    "Enter prompts (one per line):",
                    value="Once upon a time\nIn a world where\nThe future holds\nTechnology has changed\nLove is",
                    height=150
                )
                prompts = [p.strip() for p in prompt_text.split('\n') if p.strip()]
            
            elif input_method == "Synthetic Dataset":
                num_samples = st.slider("Number of samples", 5, 50, 10)
                if st.button("Generate Synthetic Dataset"):
                    dataset = create_synthetic_dataset(num_samples)
                    prompts = [sample["prompt"] for sample in dataset]
                    st.success(f"Generated {len(prompts)} synthetic prompts")
            
            elif input_method == "Upload File":
                uploaded_file = st.file_uploader("Upload JSON file with prompts", type=['json'])
                if uploaded_file:
                    try:
                        data = json.load(uploaded_file)
                        if isinstance(data, list):
                            prompts = [item.get('prompt', str(item)) for item in data]
                        elif isinstance(data, dict) and 'prompts' in data:
                            prompts = data['prompts']
                        st.success(f"Loaded {len(prompts)} prompts from file")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            
            if prompts and st.button("🚀 Generate Batch"):
                with st.spinner(f"Generating texts for {len(prompts)} prompts..."):
                    config = GenerationConfig(
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=1
                    )
                    
                    try:
                        generated_texts = st.session_state.generator.generate_multiple_prompts(prompts, config)
                        st.session_state.generated_texts = generated_texts
                        
                        # Display results
                        results_df = pd.DataFrame({
                            'Prompt': prompts,
                            'Generated Text': generated_texts
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="generated_texts.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error in batch generation: {e}")
    
    with tab3:
        st.header("Text Evaluation")
        
        if st.session_state.generated_texts:
            st.subheader("Evaluation Metrics")
            
            try:
                metrics = st.session_state.generator.evaluate_generation(st.session_state.generated_texts)
                st.session_state.metrics = metrics
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Length", f"{metrics.get('avg_length', 0):.1f} words")
                    st.metric("Average Characters", f"{metrics.get('avg_chars', 0):.0f}")
                
                with col2:
                    st.metric("Vocabulary Diversity", f"{metrics.get('vocabulary_diversity', 0):.3f}")
                    if 'perplexity' in metrics:
                        st.metric("Perplexity", f"{metrics['perplexity']:.2f}")
                
                with col3:
                    st.metric("Total Texts", len(st.session_state.generated_texts))
                    st.metric("Model", st.session_state.generator.model_name)
                
                # Visualization
                st.subheader("Text Length Distribution")
                lengths = [len(text.split()) for text in st.session_state.generated_texts]
                st.bar_chart(pd.DataFrame({'Word Count': lengths}))
                
            except Exception as e:
                st.error(f"Error computing metrics: {e}")
        else:
            st.info("Generate some texts first to see evaluation metrics.")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### 🤖 Text Generation with Transformers
        
        This application demonstrates modern text generation using Hugging Face transformers.
        
        **Features:**
        - Multiple GPT-2 model variants
        - Configurable generation parameters
        - Batch processing capabilities
        - Evaluation metrics
        - Interactive web interface
        
        **Models Available:**
        - GPT-2 (Small): Fast, good for experimentation
        - GPT-2 Medium: Balanced performance and speed
        - GPT-2 Large: Higher quality, slower
        - DistilGPT-2: Lightweight version
        
        **Generation Parameters:**
        - **Temperature**: Controls randomness (lower = more focused)
        - **Top-p**: Nucleus sampling threshold
        - **Top-k**: Limit vocabulary to top-k tokens
        - **Max Length**: Maximum length of generated text
        
        **Built with:**
        - Streamlit for the web interface
        - Hugging Face Transformers
        - PyTorch for model inference
        - Pandas for data handling
        """)
        
        st.subheader("📊 Model Information")
        if st.session_state.generator:
            st.info(f"**Current Model:** {st.session_state.generator.model_name}")
            st.info(f"**Device:** {st.session_state.generator.device}")
        else:
            st.warning("No model loaded")

if __name__ == "__main__":
    main()
