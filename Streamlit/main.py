import streamlit as st
from VeraCity_engine import *
import pickle

# Unpickle file - loading its contents
# use streamlit's decorator st.cache_resource to prevent reloading all contents of pickle file everytime
# upon every small change (uploading a file, change in code) streamlit re-runs the entire script of python, so everytime a file is uploaded it will rerun the entire main.py
# st.cache_resource prevents that, once this is loaded, saves it, and does NOT rerun this part every time

@st.cache_resource
def load_veracity_bundle():
  with open('VeraCity_model.pkl', 'rb') as file:
    return pickle.load(file)
  
veracity_bundle = load_veracity_bundle()

st.set_page_config( page_title = 'VeraCity - News Analyzer')

st.title("VeraCity - News Analyzer & Scorer")

uploaded_file = st.file_uploader('Choose a file', type = "txt")

if uploaded_file is not None:
  # Convert uploaded file to string
  article_text = uploaded_file.read().decode('UTF-8')

  with st.spinner('Analyzing...'):
    fake_likelihood, bias_score, reliability, tone_label, top_fake_triggers, top_real_indicators = news_analysis_engine(article_text, veracity_bundle)
    # Process the text by passing to news_analysis_engine function; also pass the pickle file that was just loaded
    st.success('Analysis Complete')

# -------------------------------------------------------------------------------------------------
  # displaying results
  st.subheader('Results of Analysis\n')
  col1,col2,col3 = st.columns(3)

  # 1. ML PREDICTION 
  col1.metric(
      label="Fake Likelihood", 
      value=f"{fake_likelihood:.2f}%",
      help="Probability that the content matches known misinformation patterns."
  )

  # 2. LINGUISTIC CONTEXT - BIAS & TONE
  col2.metric(
      label="Bias Level", 
      value=f"{bias_score:.2f}/100",
      help="Measures subjective and emotional language compared to Real news benchmarks."
  )

  # 3. RELIABILITY
  col3.metric(
      label="Reliability Score", 
      value=f"{reliability:.2f}%",
      help="The overall confidence rating of the article's integrity.")
  

  # Logical Color Mapping
  st.markdown("---")
  st.subheader("Tone Analysis")

  # Mapping your Step 10 & 11 logic to UI visuals
  if tone_label == 'Neutral / Standard Reporting':
      st.success(f"**Tone:** {tone_label}")
      st.caption("This text follows standards of objective journalism with minimal emotional language.")

  elif tone_label == 'Subtle Opinions and maybe Persuasive':
      st.info(f"**Tone:** {tone_label}")
      st.caption("The text includes personal viewpoints or persuasive words, but language remains calm and professional.")

  elif tone_label == 'Dramatic / Sensationalist news':
      st.warning(f"**Tone:** {tone_label}")
      st.caption("Warning: This article uses loud, emotional, or exaggerated language designed to provoke a reaction.")

  elif tone_label == 'Highly Opinionated and Emotional':
      st.error(f"**Tone:** {tone_label}")
      st.caption("High Risk: This text is heavily driven by personal bias and intense emotion rather than neutral reporting.")
  
  st.markdown("---")
  st.subheader("Top 5 real-news Indicators")
  st.write(top_real_indicators)

  st.subheader("Top 5 fake Triggers")
  st.write(top_fake_triggers)