import streamlit as st
import requests
import time

# Constants
HF_API_URL = "https://api-inference.huggingface.co/models/unsloth/Llama-3.2-1B-bnb-4bit"
HF_API_TOKEN = "your_huggingface_token"  # Replace with your actual Hugging Face API token

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Function to generate text
def generate_text(prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        },
        "options": {"wait_for_model": True},
    }

    for _ in range(5):  # Retry up to 5 times
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        elif response.status_code == 503:
            st.warning("Model is still loading. Retrying in 10 seconds...")
            time.sleep(10)  # Wait for 10 seconds before retrying
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None

    st.error("Failed to generate text after multiple attempts.")
    return None

# Streamlit UI
st.title("Hugging Face Text Generation with Llama Model")
st.write("Generate text using the `unsloth/Llama-3.2-1B-bnb-4bit` model hosted on Hugging Face.")

# Input for user prompt
prompt = st.text_area("Enter your prompt:", placeholder="Type something...")

# Model parameters
with st.sidebar:
    st.header("Model Parameters")
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=200, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_k = st.slider("Top-k Sampling", min_value=0, max_value=100, value=50, step=10)
    top_p = st.slider("Top-p Sampling", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# Generate button
if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating text..."):
            output = generate_text(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            if output:
                st.subheader("Generated Text")
                st.write(output)
    else:
        st.warning("Please enter a prompt to generate text.")
