from legal_translate import translate

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma, chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from legal_content_ingest import uploadTemp 
import streamlit as st
import os
import re

dir_path = './tempDir'

st.title("💬 பேட்டை: தமிழ் சட்ட உதவியாளர் - PettAI: Food legal assistant")



def isPDF(fileName: str):
    return fileName.endswith('pdf')



_template = """Assume you are legal assistant who helps food related laws in India. 
Help the user by explaining the answer

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
* List the most important top 5 points that are in the context
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    print(docs)
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def initVectorRetriver(file):
    vectorstore = chroma.Chroma(
        collection_name="rag-chroma",
        persist_directory=f"./tempDir/{file.name}_embed",
        embedding_function=OllamaEmbeddings(),
    )
    retriever = vectorstore.as_retriever(k=5)
    return retriever

def createChain(file):
    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOllama(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        | StrOutputParser(),
    )
    retriever = initVectorRetriver(file)
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = (_inputs | _context | ANSWER_PROMPT 
                            #    | ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                               | ChatOllama()
                            )

    chain = conversational_qa_chain
    return chain;


def identify_translate(text):
    prompt = ChatPromptTemplate.from_template("Identify the langauge of the text below. Remeber just answer in 1 word of\n1. Tamil\n2.English\3. None. {text}")
    x = prompt | ChatOllama() | StrOutputParser()
    lang = x.invoke({"text": text})
    if 'Tamil' in lang:
        out = translate(text, True)
        print('inside tamil', out)
        return out
    return text

    




if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt_str = st.chat_input()

name = ""

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")

    value = st.selectbox(
    "Choose exisiting files",
    filter(isPDF, os.listdir(dir_path)),
    index=0,
    placeholder="Select contact method...",
    )
    



class Wrapper:
    name: str
    def __init__(self, nameV):
        self.name = nameV

if uploaded_file is not None:
    # uploaded_file
    uploadTemp(uploaded_file)
else:
    uploaded_file = Wrapper(nameV=value)

chain = createChain(uploaded_file)

# text_container = st.empty()
# for i in range(10):
#     text_container.write(f"Chunk {i+1} of streaming data")

if prompt_str and chain:
    st.session_state.messages.append({"role": "user", "content": prompt_str})
    st.chat_message("user").write(prompt_str)
    translated = identify_translate(prompt_str)
    print("\nLanguage translated\n", translated)
    with st.progress(value=50, text="Generating response") as t:
        st.text("Generating response")
    msg1 = chain.invoke(
        {
            "question": translated,
            "chat_history": [],
        }
    ).content    
    
    with st.chat_message("assistant"):
        messages = []
        for i in re.split(r'\n', msg1):
            t = translate(i)
            messages.append(t)
            st.write(t)
        st.session_state.messages.append({"role": "assistant", "content": "\n".join(messages)})
        
        

    
