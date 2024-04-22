from legal_translate import translate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma, chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from operator import itemgetter
from legal_content_ingest import uploadTemp 
import streamlit as st
import os
import re
import subprocess
from content_extraction import extract_content

#Instead of a centered layout, this gives more space to view the text output.
st.set_page_config(layout="wide")

#This variable is used to select the local LLM model
model = None

#In order to view the intermediate output of LLM, a callback manager is created.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#Directory where the PDF files and embeddings are saved.
dir_path = r'./tempDir/'

st.title("💬 பேட்டை: தமிழ் சட்ட உதவியாளர் - PettAI: Food legal assistant")
tab1, tab2 = st.tabs(["Q&A section", "Summarizer"])

#To pass the previous messages as history to the LLM
qa_history_llm = ChatMessageHistory(messages=[])
sum_history_llm = ChatMessageHistory(messages=[])

# Function to get the list of available LLMs from Ollama
def get_available_models():
    res = []
    # Run the ollama list command and capture the output
    output = subprocess.check_output(["ollama", "list"])

    # Split the output into lines
    lines = output.decode().split('\n')

    if(len(lines)) == 1:
        res = []
    # Iterate through the lines of the output
    for line in lines[2:]:
        columns = line.split('\t')
        llm_name = columns[0].strip()
        if llm_name != '':
            res.append(llm_name)

    return res

def isPDF(fileName: str):
    return fileName.endswith('pdf')



_template = """ 
Help the user by explaining the answer

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """
Answer the question using the provided context.
Question: {question}

Context: {context}

Answer: 
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    print(docs)
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def classify_chat(text):
    prompt = ChatPromptTemplate.from_template("""Assume you are a classifier that classifies the incoming message whether it needs more context or not to proceed. Answer in just 1 word
Available options:
1. Answer 
2. Context

Examples:
Message: Tell me a joke
Reply: Answer

Message: Tell me about milk
Reply: Context
                                              
Message: Need information about milk
Reply: Context

Message: Hello
Reply: Answer
                                              
Message: Need more information
Reply: Context
                                              
Message: My Name is Alex
Reply: Answer
                                              
Message: My Name is Damo 
Reply: Answer
                                              
Message: My Name is 
Reply: Answer
                                              
Message: Need more information
Reply: Context
                                              
Message: Summarize the content
Reply: Answer
                                              
Message: Need more information
Reply: Context
                                              
Message: What is my name ?
Reply: Answer
                                              
Message: My name is Alex
Reply: Answer
                                            
                                     
Message: {text}""")
    x = prompt | ChatOllama(model='phi:latest',callbacks=callback_manager) | StrOutputParser()
    return x.invoke({"text": text})

def initVectorRetriver(file):
    vectorstore = chroma.Chroma(
        collection_name="rag-chroma",
        persist_directory=f"./tempDir/{file.name}_embed",
        embedding_function=OllamaEmbeddings(model="snowflake-arctic-embed"),
    )
    retriever = vectorstore.as_retriever(k=1)
    return retriever

def createChain(file):
    retriever = initVectorRetriver(file)
    rag_chain = (
        {
            "context": retriever | _combine_documents, 
            "question": RunnablePassthrough()
        }
        | ANSWER_PROMPT
        | ChatOllama(model=f'{model}',callbacks=callback_manager,temperature=0)
        | StrOutputParser()
    )
    return rag_chain
    


def createRawChain():
    contextualize_q_system_prompt = """
    You are a helpful assistant. Answer the user's query. Make user to follow user's instructions accurately. Don't deviate out of the user's instructions.
    """

    llm = ChatOllama(model=f'{model}',callbacks=callback_manager)
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    return contextualize_q_chain

def identify_translate(text):
    prompt = ChatPromptTemplate.from_template("Identify the langauge of the text below. Remeber just answer in 1 word of\n1. Tamil\n2.English\3. None. {text}")
    x = prompt | ChatOllama(model=f'{model}',callbacks=callback_manager) | StrOutputParser()
    lang = x.invoke({"text": text})
    if 'Tamil' in lang:
        out = translate(text, True)
        print('inside tamil', out)
        return out
    return text

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")

    value = st.selectbox(
    "Choose exisiting files",
    filter(isPDF, os.listdir(dir_path)),
    index=0,
    placeholder="Select contact method...",
    )
    # Get the list of available models
    models = get_available_models()
    model = st.selectbox("Select a Language Model", models, index=0)
    
while(True):
    if value is not None or uploaded_file is not None:
        break
    else:
        st.warning("Please upload a file or select an existing file to continue")
        st.stop()


class Wrapper:
    name: str
    def __init__(self, nameV):
        self.name = nameV

if uploaded_file is not None:
    # uploaded_file
    uploadTemp(uploaded_file)
else:
    uploaded_file = Wrapper(nameV=value)







# text_container = st.empty()
# for i in range(10):
#     text_container.write(f"Chunk {i+1} of streaming data")
# Create two tabs with headings

if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

if "sum_history" not in st.session_state:
    st.session_state["sum_history"] = []

with tab1:
    qa_messages = st.container(height=325, border=False)

    prompt_str_tab1 = st.chat_input('Enter your query',key='tab1')

    for msg in st.session_state.qa_history:
        qa_messages.chat_message(msg["role"]).markdown(msg["content"])

    if prompt_str_tab1:
        qa_messages.chat_message("user").markdown(prompt_str_tab1)
        st.session_state.qa_history.append({"role": "user", "content": prompt_str_tab1})
        # translated = identify_translate(prompt_str)
        translated = prompt_str_tab1
        print("\nLanguage translated\n", translated)
        # with st.progress(value=50, text="Generating response") as t:
        #     st.text("Generating response")
        
        # checkpoint = classify_chat(translated)
        # print("checkpoint", checkpoint)
        checkpoint = 'Context'

        if checkpoint.startswith('Answer'):
            plain_chain = createRawChain()
            msg1 = plain_chain.invoke(
                {
                    "question": translated,
                    "chat_history": qa_history_llm.messages
                }
            )
        else:
            chain = createChain(uploaded_file)
            msg1 = chain.invoke(
                {
                    "question": translated,
                    "chat_history": qa_history_llm.messages
                }
            )

        # msg1 = 'You are awesome!'

        qa_history_llm.add_user_message(translated)
        qa_history_llm.add_ai_message(msg1)
        # messages = []
        # for i in re.split(r'\n', msg1):
        #     if not i == '' or i == '\n':
        #         t = translate(i.strip().replace('. ', ''))
        #         messages.append(t.replace('. ', ''))
        # st.session_state.qa_history.append({"role": "assistant", "content": "\n".join(messages)})
        qa_messages.chat_message("assistant").markdown(msg1)
        st.session_state.qa_history.append({"role": "assistant", "content": msg1})
        print(st.session_state["qa_history"])

with tab2:
    sum_messages = st.container(height=325, border=False)
    prompt_str_tab2 = st.chat_input('Enter the heading to summarize')
    for msg in st.session_state.sum_history:
            sum_messages.chat_message(msg["role"]).markdown(msg["content"])
    if prompt_str_tab2:
        sum_messages.chat_message("user").markdown(prompt_str_tab2)
        st.session_state.sum_history.append({"role": "user", "content": prompt_str_tab2})
        # with st.progress(value=50, text="Generating response") as t:
        #     st.text("Generating response")
        question = "Summarize the following content:"
        content,pgs = extract_content(uploaded_file.name, prompt_str_tab2)
        # st.write(content)
        question += f'\n{content}'

        # plain_chain = createRawChain()
        # msg1 = plain_chain.invoke(
        #     {
        #         "question": question,
        #         "chat_history": sum_history_llm.messages
        #     }
        # )
        # msg1 += f'\nThe content spanned page(s): {",".join(pgs)}'

        msg1 = content

        sum_history_llm.add_user_message(question)
        sum_history_llm.add_ai_message(msg1)
        # messages = []
        # for i in re.split(r'\n', msg1):
        #     if not i == '' or i == '\n':
        #         t = translate(i.strip().replace('. ', ''))
        #         messages.append(t.replace('. ', ''))
        # st.session_state.sum_messages.append({"role": "assistant", "content": "\n".join(messages)})
        st.session_state.sum_history.append({"role": "assistant", "content": msg1})
        print(st.session_state["sum_history"])
        sum_messages.chat_message("assistant").markdown(msg1)