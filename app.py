from flask import Flask, render_template, jsonify, request
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeSparseVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Get Pinecone API key from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Get GitHub Token from environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Define the GitHub AI model inference endpoint and model name
GITHUB_AI_ENDPOINT = "https://models.github.ai/inference"
# Explicitly setting the model name to Phi-3-small-8k-instruct
GITHUB_AI_MODEL_NAME = "Phi-3-small-8k-instruct"

# Set environment variables for Pinecone (if not already set globally)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

decsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

retriever = decsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(
    temperature=0.4,
    max_tokens=500,
    model=GITHUB_AI_MODEL_NAME, # This is now "Phi-3-small-8k-instruct"
    openai_api_key=GITHUB_TOKEN,
    openai_api_base=GITHUB_AI_ENDPOINT
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sytem_prompt),
        ("human", "{input}")
    ]
)

chain = create_stuff_documents_chain(llm,prompt)
rag = create_retriever_chain(
    retriever,chain)

@app.route('/')
def index():
    return render_template('chat.html')
@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("Response : ",response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)