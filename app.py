import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load model and tokenizer
# -------------------------------

model = tf.keras.models.load_model("next_word_lstm.h5")# TensorFlow SavedModel folder

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Determine maximum sequence length from model
max_sequence_len = model.input_shape[1] + 1

# -------------------------------
# Predict next word function
# -------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
    return reverse_word_index.get(predicted_word_index, "")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Next Word Prediction App")
st.write("Type a sentence and the app will predict the next word.")

# Text input
user_input = st.text_input("Enter text:")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        if next_word:
            st.success(f"Predicted next word: **{next_word}**")
        else:
            st.info("Could not predict the next word. Try a different input.")
