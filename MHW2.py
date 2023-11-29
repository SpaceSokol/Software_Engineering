from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from sacremoses import MosesTokenizer, MosesDetokenizer
#from tensorflow.contrib.keras.preprocessing.text import Tokenizer
import io

import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-ru-en")

#st.title = ("Перевод текста с русского на английский")
my_input = st.text_input("ТЕКСТ ДЛЯ ПЕРЕВОДА")
result = st.button("ПЕРЕВОД ТЕКСТА")
if result:
    input_ids = tokenizer.encode(my_input, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(decoded_output)

