import joblib
import streamlit as st
import pdfplumber as plu
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load precomputed data
df = joblib.load("data.pkl")
vector = joblib.load("feature_matrix.pkl")
vec = joblib.load("vectorizer.pkl")

dfc=df.copy()

# Function to process uploaded resume
def file_resume (file):
    with plu.open(file) as pdf:
        page=pdf.pages[0]
        text=page.extract_text()
        par=text.replace("\n"," ")
    rvec=vec.transform([par]).toarray()
    sim=cosine_similarity(rvec,vector)[0]
    df['sim']=sim
    df_new=df.sort_values(by='sim',ascending=False).head()
    lis=df_new['Position'].unique().tolist()
    st.write("\n".join([f"{i+1} {lis[i]}" for i in range(min(3, len(lis)))]))

#####
# Tabs
tab1, tab2 = st.tabs(['Position Recommender', 'Resume Selector'])

with tab1:
    st.title("Resume Recommendation Website")

# File uploader
    file = st.file_uploader("Upload Your Resume", type=['pdf'])

    if file is not None:
        file_resume(file)
    else:
        st.info("Please upload a PDF resume to see recommendations.")

with tab2:
    st.title("📂 Resume selector")

    # Upload multiple PDFs
    folder = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

    Positions=df['Position'].unique().tolist()
    pos=st.selectbox("Select Position for which you have to select the resume",Positions)
    box=st.checkbox("Plese click if all files are uploaded")

    #######
    if pos:
        if box:
            lis=[]
            for file in folder:
                with plu.open(file) as pdf:
                    page=pdf.pages[0]
                    text=page.extract_text()
                    par=text.replace("\n"," ")
                    rvec=vec.transform([par]).toarray()
                    rsim=cosine_similarity(rvec,vector)[0]
                    df[file.name]=rsim
                    lis.append(file.name)
            df_pos=df[df['Position']==pos]       
            max_lis=df_pos[lis].idxmax(axis=1)
            result= max_lis.value_counts().idxmax()
            st.write(f"Congragulation {result} is selected for {pos} Position")
