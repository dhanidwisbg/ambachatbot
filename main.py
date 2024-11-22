import streamlit as st
import requests

# Konfigurasi API Hugging Face
HF_API_URL = "https://api-inference.huggingface.co/models/joshmiller656/Llama3.2-1B-AWQ-INT4"  # Model URL
HF_API_TOKEN = "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"  # Masukkan token Anda di sini

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_huggingface_api(prompt):
    """
    Kirim prompt ke Hugging Face API dan ambil responnya.
    """
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Aplikasi Streamlit
st.title("AMBACHATBOT")
st.write("Aplikasi chatbot berbasis model mas rusdi.")

# Input prompt dari user
user_input = st.text_input("Masukkan Prompt Anda:", "")

if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Menghasilkan respons..."):
            response = query_huggingface_api(user_input)
        st.write("### Respon:")
        st.write(response)
    else:
        st.error("Prompt tidak boleh kosong!")
