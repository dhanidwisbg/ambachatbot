from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

# Path to your local model directory
model_download_dir = "./model_download_dir"  # Path to the folder where model is stored

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_download_dir)
model = GPT2LMHeadModel.from_pretrained(model_download_dir)

# Streamlit UI
st.title("Text Generation with Local GPT-2 Model")
st.write("Generate text using a local GPT-2 model.")

# Input for user prompt
prompt = st.text_area("Enter your prompt:", placeholder="Type something...")

# Model parameters
with st.sidebar:
    st.header("Model Parameters")
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=200, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_k = st.slider("Top-k Sampling", min_value=0, max_value=100, value=50, step=10)
    top_p = st.slider("Top-p Sampling", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# Function to generate text from the local model
def generate_text(prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    # Encode the prompt to get input IDs
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text using the local model
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_tokens + len(inputs[0]),  # Adjust max length for the prompt length
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Avoid repetition
        )
    
    # Decode the generated output to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

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
