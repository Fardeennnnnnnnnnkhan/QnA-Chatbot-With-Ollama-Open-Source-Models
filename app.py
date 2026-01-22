import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
load_dotenv()

#LangSmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']='Simple QnA Chatbot With OLLAMA'

##Prompt Template 
prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a helpful assistant. Please response ot the user queries"),
    ("user","Question : {question}")
])

def generate_response(question,engine ,temperature,max_tokens):
    llm = OllamaLLM(model = engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the APP
st.title("Enhanced Q&A Chatbot with Ollama")

## Sidebar for settings
st.sidebar.title("Settings")

##DropDown for selecting various openAI Models
engine = st.sidebar.selectbox("Select an Open AI Model", ['gemma:2b','mistral'])

#Adjust response Paramater -> Temparatures and Max_tokens
temperature =st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0 , value = 0.7)
max_tokens =st.sidebar.slider("Max Tokens",min_value=50, max_value=300, value = 150)

## Main Interface for user Input
st.write("Go Ahead and ask any question")
user_input=st.text_input("You: ")

if user_input:
    response = generate_response(user_input , engine  ,temperature,max_tokens)
    st.write(response)
else:
    st.write("Provide a Query ! ")
    