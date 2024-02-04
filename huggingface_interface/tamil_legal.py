# from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings, GPT4AllEmbeddings, LocalAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import  chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
import os
from langchain.vectorstores import Chroma, chroma
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator

class Embed4All:
    """
    Python class that handles embeddings for GPT4All.
    """

    def __init__(self, model_name: Optional[str] = None, n_threads: Optional[int] = None, **kwargs):
        """
        Constructor

        Args:
            n_threads: number of CPU threads used by GPT4All. Default is None, then the number of threads are determined automatically.
        """
        import fasttext
        self.gpt4all = fasttext.load_model("/Users/damodharan-2579/Downloads/cc.ta.300.bin")
        

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding.

        Args:
            text: The text document to generate an embedding for.

        Returns:
            An embedding of your document of text.
        """
        # return self.gpt4all.model.generate_embedding(text)
        return self.gpt4all.get_sentence_vector(text)

class TamilEmbedding(BaseModel, Embeddings):
    client: Any  #: :meta private:

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that GPT4All library is installed."""

        try:
            from gpt4all import Embed4All

            values["client"] = Embed4All()
        except ImportError:
            raise ImportError(
                "Could not import gpt4all library. "
                "Please install the gpt4all library to "
                "use this embedding model: pip install gpt4all"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using GPT4All.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        embeddings = [self.client.embed(text) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using GPT4All.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

file = "/Users/damodharan-2579/legal-chat/translate-module/IndicTrans2/huggingface_interface/ponni.txt"

def uploadTemp():
    # try:
    #     with open(f"./tempDir/{file.name}", "r") as f:
    #         # Embedding already exisits
    #         return
    # except:    
        # create embeddings
    # with open(file,"wb") as f:
    #     f.write(file.getbuffer())
    loader = TextLoader(file)
    data = loader.load()
    # print(data[0])
    #Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=[". "])
    all_splits = text_splitter.split_documents(data)

    vectorstore = chroma.Chroma.from_documents(documents=all_splits, 
                                        collection_name="rag-chroma",
                                        embedding=TamilEmbedding(),
                                        persist_directory=f"./ponni/"
                                        )
    vectorstore.persist()

def initVectorRetriver():
    vectorstore = chroma.Chroma(
        collection_name="rag-chroma",
        persist_directory=f"./ponni/",
        embedding_function=GPT4AllEmbeddings(),
    )
    retriever = vectorstore.as_retriever(k=5)
    return retriever



print(initVectorRetriver().get_relevant_documents(query="சிங்கம்"))
# pdf2Text()
    
# uploadTemp()