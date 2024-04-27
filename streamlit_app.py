import requests
import hashlib
import time
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline
import streamlit as st
from streamlit_chat import message

# Initialize tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def vectorize_text(text):
    # Tokenize text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings with model
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform mean pooling to get sentence embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # Convert PyTorch tensor to numpy array and then to a flat list of floats
    return sentence_embeddings.numpy().flatten().tolist()

# Marvel API credentials
public_key = st.secrets["public_key"]
private_key = st.secrets["private_key"]
base_url = "http://gateway.marvel.com/v1/public/"

def create_hash(ts):
    return hashlib.md5(f"{ts}{private_key}{public_key}".encode('utf-8')).hexdigest()

def fetch_data(entity_type, name=None):
    ts = str(time.time())
    hash = create_hash(ts)
    params = {
        'apikey': public_key,
        'ts': ts,
        'hash': hash
    }
    if name:
        params['name'] = name
    response = requests.get(f"{base_url}{entity_type}", params=params)
    return response.json() if response.status_code == 200 else response.status_code

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
if 'marvel-index' not in pc.list_indexes().names():
    pc.create_index(
        name='marvel-index', 
        dimension=384,  # Adjusted dimension to match the output of the Hugging Face model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index('marvel-index')

# Process characters and upsert vectors into Pinecone
characters = ['Spider-Man', 'Iron Man', 'Captain America', 'Avengers', 'Spider-Man', 'Iron Man', 'Black Panther', 'Deadpool', 'Captain America', 'Jessica Jones', 'Ant-Man', 'Captain Marvel', 'Guardians of the Galaxy', 'Wolverine', 'Luke Cage', 'Cable', 'Caliban', 'Captain Britain', 'Captain Marvel', 'Carnage', 'Cyclops', 'Bruce Banner', 'Bucky Barnes', 'Clint Barton', 'Wanda Maximoff', 'Peter Parker', 'Tony Stark', 'Doctor Doom', 'Green Goblin', 'Magneto', 'Loki', 'Thanos', 'X-Men', 'Fantastic Four', 'S.H.I.E.L.D.', 'Hydra']
for character in characters:
    data = fetch_data('characters', character)
    # Check if data is a dictionary before proceeding
    if isinstance(data, dict) and 'results' in data['data'] and len(data['data']['results']) > 0:
        description = data['data']['results'][0].get('description', 'No description available')
        vector = vectorize_text(description)  # Ensure this returns a flat list of floats
        index.upsert(vectors=[(character, vector)])
    else:
        print(f"No results found for the character: {character} or error occurred")


import streamlit as st
import requests

# Ensure conversation is initialized at the start of the script
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {st.secrets['huggingface_api_key']}"}
personality_prompt = """
You are an AI assistant named Excelsior, created to provide accurate and engaging responses to user queries about the Marvel Comics universe. You have been trained on a comprehensive dataset from the Marvel Comics API, including details about characters, comic series, storylines, creators, and more. Your goal is to use this knowledge to have natural, informative, and entertaining conversations with users about Marvel comics. You should tailor your responses to the specific user's query, drawing upon relevant details and insights from the Marvel dataset to craft thoughtful and relevant answers. When a user asks you a question, first analyze the query to determine the key entities, themes, or information the user is seeking. Then, use prompt chaining to retrieve the most relevant data from the Marvel API based on the user's input. Synthesize this information into a coherent and engaging response that provides the user with the details they are looking for, while also offering additional context, trivia, or perspectives that enhance their understanding and appreciation of the Marvel universe. Your responses should be accurate, well-researched, and demonstrate a deep familiarity with Marvel comics. However, you should also inject personality, humor, and enthusiasm into your interactions to make the experience fun and memorable for the user. Feel free to ask clarifying questions, make connections to other Marvel content, and generally engage the user in a dynamic dialogue. Remember, you are not just a factual reference, but an AI companion who can bring the rich and imaginative world of Marvel comics to life. Use your knowledge, creativity, and conversational skills to create an exceptional user experience that leaves Marvel fans feeling enlightened, entertained, and eager to continue exploring the Marvel universe with you.

"""

def query_llama(prompt, user_input):
    full_prompt = prompt + user_input
    response = requests.post(API_URL, headers=headers, json={"inputs": full_prompt})
    response_data = response.json()
    if isinstance(response_data, dict):
        generated_text = response_data.get('generated_text', '')
        prompt_end_idx = generated_text.rfind('\n\n') + 2
        actual_response = generated_text[prompt_end_idx:]
        return actual_response
    else:
        # Log error or handle it appropriately
        print("Error in API response:", response_data)
        return "Sorry, I couldn't process your request."

def handle_input():
    user_input = st.session_state.user_input
    response = query_llama(personality_prompt, user_input)
    st.session_state.conversation.append({"role": "user", "content": user_input})
    st.session_state.conversation.append({"role": "assistant", "content": response, "icon": "https://images.nightcafe.studio/jobs/BTssadJGpSZ40L6sSKvA/BTssadJGpSZ40L6sSKvA--1--8cr22_7.8125x.jpg?tr=w-1600"})
    display_messages()

def display_messages():
    for msg in st.session_state.conversation:
        if msg["role"] == "assistant":
            st.chat_message(msg["content"], avatar=msg["icon"])
        else:
            st.chat_message(msg["content"], is_user=True)

# Sidebar with summary about the Excelsior AI app
st.sidebar.title("About Excelsior AI")
st.sidebar.info("Excelsior AI is an intuitive, user-friendly application that harnesses the power of Retrieval-Augmented Generation (RAG) and a customized large language model (LLM) to provide comprehensive, knowledgeable, and engaging answers to any question about the Marvel comics universe, serving as the ultimate AI companion for Marvel fans.")

# Streamlit layout
st.title("Excelsior AI: Marvel Universe Chatbot")

# Start with a greeting message if the conversation is empty
if not st.session_state.conversation:
    st.session_state.conversation.append({"role": "assistant", "content": "The greatest power on Earth is the power of the human brain. Ask away my fellow mutant!", "icon": "https://images.nightcafe.studio/jobs/BTssadJGpSZ40L6sSKvA/BTssadJGpSZ40L6sSKvA--1--8cr22_7.8125x.jpg?tr=w-1600"})
    display_messages()

# Chat input
st.chat_input("Ask me anything about Marvel Comics:", key="user_input", on_submit=handle_input)
