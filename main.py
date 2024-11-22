from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import streamlit as st

# Tentukan path ke direktori model lokal
model_download_dir = './model_download_dir'  # Pastikan path ini benar

# Memuat tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_download_dir)
model = GPT2LMHeadModel.from_pretrained(model_download_dir)

# UI Streamlit
st.title("Generasi Teks dengan Model GPT-2 Lokal")
st.write("Generate text menggunakan model GPT-2 lokal.")

# Input prompt dari pengguna
prompt = st.text_area("Masukkan prompt Anda:", placeholder="Ketik sesuatu...")

# Parameter model
with st.sidebar:
    st.header("Parameter Model")
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=200, value=100, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_k = st.slider("Top-k Sampling", min_value=0, max_value=100, value=50, step=10)
    top_p = st.slider("Top-p Sampling", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# Fungsi untuk menghasilkan teks dari model lokal
def generate_text(prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    # Mengkode prompt untuk mendapatkan input IDs
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Menghasilkan teks menggunakan model lokal
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_tokens + len(inputs[0]),  # Sesuaikan panjang max untuk panjang prompt
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Menghindari pengulangan
        )
    
    # Mendekode output yang dihasilkan menjadi teks
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Tombol untuk menghasilkan teks
if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating text..."):
            output = generate_text(prompt, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
            if output:
                st.subheader("Generated Text")
                st.write(output)
    else:
        st.warning("Please enter a prompt to generate text.")
