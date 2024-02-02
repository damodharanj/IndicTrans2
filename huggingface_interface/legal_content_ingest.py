from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings, GPT4AllEmbeddings, LocalAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import  chroma
from langchain.document_loaders import PyPDFLoader
import os

def uploadTemp(file):
    try:
        with open(f"./tempDir/{file.name}", "r") as f:
            # Embedding already exisits
            return
    except:    
        # create embeddings
        with open(os.path.join("tempDir",file.name),"wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(f"./tempDir/{file.name}")
        data = loader.load()
        
        #Split
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=[". "])
        all_splits = text_splitter.split_documents(data)

        vectorstore = chroma.Chroma.from_documents(documents=all_splits, 
                                            collection_name="rag-chroma",
                                            embedding=OllamaEmbeddings(),
                                            persist_directory=f"./tempDir/{file.name}_embed/"
                                            )
        vectorstore.persist()

    

    
    