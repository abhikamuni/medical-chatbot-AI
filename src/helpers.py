from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extract text from a PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data, 
                             glob = "*.pdf",   #LOAD ALL PDF FILES IN THE DIRECTORY
                             loader_cls=PyPDFLoader) # EXTRACT TEXT FROM PDF FILES
    documents = loader.load()
    return documents


#split the Data into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts_chunks = text_splitter.split_documents(extracted_data)
    return texts_chunks

#download the embeddings model from HuggingFace
def download_hugging_face_embeddings():
    embeddngs = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddngs