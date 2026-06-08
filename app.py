
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# UI Configuration aur Styling
st.set_page_config(page_title="Text Generator", page_icon="📝", layout="centered")

st.markdown('''
    <style>
    .main { background-color: #fafbfc; }
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; font-weight: bold; }
    h1 { color: #1E293B; }
    </style>
''', unsafe_allow_html=True)

st.title("📝 Predictive Text Generator")
st.write("Apna koi bhi phrase/word enter karein aur dekhein model aage ke words kaise predict karta hai.")

# Assets load karne ka function
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('my_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    max_seq_len = model.input_shape[1] + 1
except Exception as e:
    st.error(f"Error: 'my_model.h5' ya 'tokenizer.pickle' nahi mili. Pehle check karein ki files Colab mein uploaded hain ya nahi. Details: {e}")
    st.stop()

# Text generation logic
def predict_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if not output_word:
            break
        seed_text += " " + output_word
    return seed_text

# Inputs
seed_text = st.text_input("Enter Seed Text:", "Machine learning")
next_words = st.slider("Kitne words predict karne hain?", min_value=1, max_value=30, value=10)

if st.button("Generate Next Words"):
    if not seed_text.strip():
        st.warning("Kripya pehle kuch text likhein!")
    else:
        with st.spinner("Model soch raha hai..."):
            result = predict_text(seed_text, next_words)
            st.success("### Prediction Output:")
            st.info(result)
