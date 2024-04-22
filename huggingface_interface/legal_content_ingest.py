from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings, GPT4AllEmbeddings, LocalAIEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, chroma
from langchain_community.document_loaders import PyPDFLoader
import os

def uploadTemp(file):
    try:
        with open(f"./tempDir/{file.name}", "r") as f:
            # Embedding already exisits
            return
    except:    
        # create embeddings
        print('Creating embeddings')
        with open(os.path.join("tempDir",file.name),"wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(f"./tempDir/{file.name}")
        data = loader.load()
        from heading_extraction import main_driver
        res = None
        main_driver(f"./tempDir/{file.name}")
        #Split
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = text_splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=512,
                                            chunk_overlap=128,
                                            separators=["\n\n",
                                                "\n",
                                                ".",
                                                ","],
                                        )
        all_splits = text_splitter.split_documents(data)

        vectorstore = chroma.Chroma.from_documents(documents=all_splits, 
                                            collection_name="rag-chroma",
                                            embedding=OllamaEmbeddings(model="snowflake-arctic-embed"),
                                            persist_directory=f"./tempDir/{file.name}_embed/"
                                            )
        vectorstore.persist()
        print('Embeddings created')

    

    
    