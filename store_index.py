from src.helpers import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Define the GitHub AI model inference endpoint and model name
GITHUB_AI_ENDPOINT = "https://models.github.ai/inference"
# Explicitly setting the model name to Phi-3-small-8k-instruct
GITHUB_AI_MODEL_NAME = "Phi-3-small-8k-instruct"





extracted_data = load_pdf_file(data='Data/') #
texts_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(
    api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

pc.create_index(
    index_name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

decsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    index_name = index_name,
    embedding=embeddings,
)