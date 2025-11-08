# app_EDA.py

import os
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq  # Modern LLM

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Business Analyst", page_icon="🤖")
st.title("AI Business Analyst Assistant 🤖")
st.write("Hello 👋! Upload a CSV file and let AI analyze it for you.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.write("### About this App")
    st.caption(
        """A smart assistant that simplifies your business data analysis.
        Upload your dataset, and the AI will help explore, understand columns,
        and give insights about your data."""
    )
    st.divider()
    st.caption("💡 Made with love by **Chandriya**")

    with st.expander("📘 What are the steps of EDA?"):
        st.write("""
        1. Understand the data structure  
        2. Check for missing values  
        3. Identify data types  
        4. Explore summary statistics  
        5. Visualize distributions  
        6. Detect outliers and anomalies  
        7. Prepare data for modeling
        """)

# -------------------- FILE UPLOAD --------------------
user_CSV = st.file_uploader("📂 Upload your CSV file", type="csv")

if user_CSV is not None:
    df = pd.read_csv(user_CSV, low_memory=False)
    st.success("✅ File uploaded successfully!")
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # -------------------- INITIALIZE LLM --------------------
    st.write("🔹 Initializing AI model...")
    GROQ_API_KEY = "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-"  # 👈 Replace this with your actual key
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b",
        temperature=0
    )

    # -------------------- CREATE AI AGENT --------------------
    st.write("🤖 Creating AI Pandas Agent...")
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True  # Required for pandas code execution
    )

    # -------------------- QUESTION INPUT --------------------
    st.subheader("💬 Ask a question about your data")
    question = st.text_input("Type your question below 👇", "What are the columns about?")

    if st.button("Ask AI") and question:
        with st.spinner("Analyzing your data..."):
            try:
                response = agent.run(question)
                st.success("✅ Here's what I found:")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

else:
    st.info("👆 Please upload a CSV file to begin.")

