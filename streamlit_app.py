import streamlit as st
import torch
import time
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

 #########
# SIDEBAR #
 #########
col1, mid, col2 = st.sidebar.columns([1, 4, 1])
mid.write('images/logo.png, width=200')
st.sidebar.markdown("<h3 style='text-align: center;font-family:Ubuntu; font-size:20px;'<p>AI SUMMARIZATION</p></h3>", unsafe_allow_html=True)
nav = st.sidebar.radio('', ['START', 'DEMO 1', 'DEMO 2', 'more'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

# CONTACT
########

expander = st.sidebar.expander('Contact')
col1, mid, col2 = expander.columns(3)
col1.write('images/_the_famous_boy.png, width=60')
mid.write("<h3 style='text-align: left;font-family:Ubuntu; font-size:10px;'<p>lalala</p></h3>", unsafe_allow_html=True)

if nav == 'START':
    
    st.markdown("<p style='text-align: center;font-family:Ubuntu;color:hsl(194, 85%, 46%); font-size:30px;'>Demo</p>",unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;font-family:Ubuntu;color:hsl(194, 85%, 25%); font-size:56px;'<p>Absolutorcinus</p></h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;font-family:Ubuntu;color:hsl(194, 85%, 46%); font-size:30px;'>Innovation</p>",unsafe_allow_html=True)
    
 

if nav == 'DEMO 1':
  
    st.markdown(f'<h3 style="text-align: left; color:hsl(194, 85%, 25%); font-size:28px;">Summarize Text</h3>',unsafe_allow_html=True)
    st.text('')

    source = st.radio("INPUT",
                      ("input bellow text", "upload  file"))
    st.text('')
    st.text('')

    s_example = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    if source == 'input bellow text':
        input_su = st.text_area(
            "Use the example below or input your own text in English (between 1,000 and 10,000 characters)",
            value=s_example, max_chars=10000, height=330)
        if st.button('Summarize'):
      
            if len(input_su) < 1000:
                #raise the warning
                st.error('Please enter a text in English of minimum 1,000 characters')
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
                tokenizer = AutoTokenizer.from_pretrained("t5-base")
                with st.spinner('Processing...'):
               
                    # T5 uses a max_length of 512 so we cut the article to 512 tokens.
                    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
                    outputs = model.generate(inputs["input_ids"], max_length=300, min_length=20, length_penalty=2.0, num_beams=4,early_stopping=True)
                    t_r = tokenizer.decode(outputs[0]) #################################################################### here I put my function
                    result_t_r = (str(len(t_r)) + ' characters' + ' ('"{:.0%}".format(
                        len(t_r) / len(input_su)) + ' of original content)')
                    print("end")
                    st.markdown('___')
                    st.write('Summary:')
                    st.caption(result_t_r)
                    st.success(t_r)
