import streamlit as st


import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate

# Marvel API key and base URL
API_KEY = "f1bc84446cbf13db579ecfd668c4cb24"
BASE_URL = "http://gateway.marvel.com/v1/public/"

# Initialize VectorStoreIndex
index = VectorStoreIndex()

# Initialize tokenizer and model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency on compatible hardware
    device_map="auto"  # Automatically use the GPU if available
)

# Function to fetch data from the Marvel API
@st.cache(suppress_st_warning=True)
def fetch_data(endpoint, params):
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()['data']['results']
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

# Function to index data into VectorStoreIndex
def index_data():
    endpoints = ["comics", "series", "stories", "events", "creators", "characters"]
    params = {
        "apikey": API_KEY,
        "limit": 350  # Adjust based on your rate limit and needs
    }

    for endpoint in endpoints:
        full_url = f"{BASE_URL}{endpoint}"
        data = fetch_data(full_url, params)
        df = pd.DataFrame(data)
        for _, row in df.iterrows():
            document = {
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "resourceURI": row.get("resourceURI", ""),
                "type": endpoint
            }
            index.add_document(document)

# Function to generate response using Llama 3 model
def generate_response(user_query, context):
    ai_persona = """
    You are an AI assistant named Excelsior, created to provide accurate and engaging responses to user queries about the Marvel Comics universe.
    """

    prompt = f"""
    {ai_persona}
    User's query: {user_query}
    Relevant context: {context}
    Response:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=250,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# Streamlit app setup
def main():
    st.set_page_config(page_title="Excelsior AI", page_icon=":sparkles:")

    st.title('Professor X')
    st.caption('Get all your questions answered about the marvel universe')

    st.sidebar.header("Excelsior AI")
    st.sidebar.write("Excelsior AI is an intuitive, user-friendly application that harnesses the power of Retrieval-Augmented Generation and a Llama 3 LLM model to provide comprehensive, knowledgeable, and engaging answers to any question about the Marvel comics universe.")

    user_query = st.text_input("Ask your question here:")
    if user_query:
        context = index.search(user_query)
        response = generate_response(user_query, context)
        response_with_attribution = response + "\n\nData provided by Marvel. © 2014 Marvel"
        st.write(response_with_attribution)

    feedback = st.radio("Was the response helpful?", ("Yes", "No"))
    if feedback == "No":
        st.text_input("Please provide feedback on how we can improve:")

if __name__ == "__main__":
    main()
