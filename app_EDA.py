import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import HuggingFaceHub

# Load environment variables
load_dotenv()
apikey = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # or any other text-generation model
    model_kwargs={"temperature":0, "max_new_tokens":256},
    huggingfacehub_api_token=apikey
)


st.title('AI Business Analyst Assistant 🤖')
st.write('Upload your CSV to analyze your data.')

user_CSV = st.file_uploader("Upload Your CSV here", type="csv")

if user_CSV is not None:
    df = pd.read_csv(user_CSV, low_memory=False)
    st.dataframe(df.head())


    # Create pandas agent
    pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Ask a question
    question = "What is the meaning of the columns?"
    response = pandas_agent.run(question)

    # Show response
    st.subheader("AI Answer:")
    st.write(response)
