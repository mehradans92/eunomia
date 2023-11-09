import streamlit as st  # Web App

# st.set_page_config(layout="wide")

import os
from PIL import Image
import asyncio
import eunomia
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from eunomia.agents import Eunomia
from eunomia.tools import EunomiaTools
from tempfile import NamedTemporaryFile
from contextlib import redirect_stdout
import time
import json
import io


# Initialize session state for logging if it's not already defined
if "log" not in st.session_state:
    st.session_state.log = ""


# Display the log in real-time
def display_log():
    with st.expander("Console Log", expanded=True):
        st.text(st.session_state.log)


text_input = ""
uploaded_file = ""

import pickle

docs = None
api_key = " "


image = Image.open("img/TOC.png")
# st.image(image, width=1000)

col1, col2, col3 = st.columns([1, 2, 3])

# Using the middle column to display the image
with col1:
    st.image(image, width=800)

# title
st.title("Agent-based Learning of Materials Datasets from Scientific Literature")
st.markdown(
    "##### This tool will allow you to extract information on water stability of metal-organic frameworks from a scientific paper or input txt. It is a agent that uses OpenAI's GPT models, and you must have your own API key. Each query is about 16k tokens, which costs about only $0.50 on your own API key, charged by OpenAI."
)

st.markdown("Used libraries:\n * [Eunomia](https://github.com/AI4ChemS/Eunomia)")


api_key_url = (
    "https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
)

api_key = st.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    help=f"['What is that?']({api_key_url})",
    type="password",
    value="",
)

os.environ["OPENAI_API_KEY"] = f"{api_key}"  #
if len(api_key) != 51:
    st.warning("Please enter a valid OpenAI API key.", icon="⚠️")


max_results_current = 5
max_results = max_results_current


def search_click_callback(search_query, max_results, XRxiv_servers=[]):
    global pdf_info, pdf_citation
    search_engines = XRxivQuery(search_query, max_results, XRxiv_servers=XRxiv_servers)
    pdf_info = search_engines.call_API()
    search_engines.download_pdf()

    return pdf_info


import streamlit as st


# Radio button for user choice between uploading a file and entering text
input_method = st.radio("Choose input method:", ("Upload PDF file", "Input text"))

uploaded_file = None
text_input = None

# Conditional logic to display the file uploader or the text input area based on the radio button selection
if input_method == "Upload PDF file":
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload a file (PDF only)", type="pdf"
    )
elif input_method == "Input text":
    text_input = st.text_area("Enter your text here:")

# A button to trigger the processing
process_button = st.button("Process Input")

if process_button:
    if input_method == "Upload PDF file" and uploaded_file is not None:
        # Process the uploaded PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        docs_processor = eunomia.LoadDoc(file_name=file_path, encoding="utf8")
        sliced_pages = docs_processor.process(
            ["references ", "acknowledgement", "acknowledgments", "references\n"],
            chunk_size=1000,
            chunk_overlap=50,
            chunking_type="fixed-size",
        )
        st.write("PDF processed.")
    elif input_method == "Input text" and text_input:
        # Process the input text
        docs_processor = eunomia.LoadDoc(text_input=text_input)
        sliced_pages = docs_processor.process(
            chunk_size=600, chunk_overlap=50, chunking_type="NLTK"
        )
        st.write("Text processed.")

    Embedding_model = "text-embedding-ada-002"
    faiss_index = FAISS.from_documents(
        sliced_pages, OpenAIEmbeddings(model=Embedding_model)
    )

    tools = EunomiaTools(
        tool_names=["read_doc", "eval_justification", "recheck_justification"],
        vectorstore=faiss_index,
    ).get_tools()
    agent = Eunomia(tools=tools, model="gpt-4", get_cost=True)

    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        with st.spinner("⏳ Please wait..."):
            start = time.time()
            result = agent.run(
                prompt="Read the document and find the MOFS and their water stability.\
                        \Check your answers and if the reasoning is not valid, try to find better reasons.\
                        Your final answer should be in a parsable JSON format."
            )
            # Pretty print the JSON with an indent of 4
            final_answer = json.loads(result)
            pretty_json = json.dumps(final_answer, indent=4)
            length_answer = len(pretty_json)
            st.text_area("Answer:", pretty_json, height=max(length_answer // 4, 200))
            end = time.time()
            clock_time = end - start
            with st.empty():
                st.write(f"✔️ Task completed in {clock_time:.2f} seconds.")


# When the user makes a selection, Streamlit will automatically rerun the script and update the display.

# # You may want to have a button to process the input, which can be placed outside of the conditional logic
# process_button = st.button("Process Input")

# # Logic to process the input based on what was entered after the user clicks the 'Process Input' button
# if process_button:
#     if input_method == "Upload PDF file" and uploaded_file is not None:
#         st.write("PDF file has been uploaded.")
#     elif input_method == "Input text":
#         st.write("Text entered:", text_input)


# Logic to handle the form data when it's submitted
# if submit_button:
#     if input_method == "Upload PDF file" and uploaded_file is not None:
#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         st.write("PDF file has been uploaded.")
#     elif input_method == "Input text":
#         st.write("Text input received:", text_input)


# with st.form(key="columns_in_form", clear_on_submit=False):
#     c1, c2 = st.columns([5, 0.8])
#     with c1:
#         search_query = st.text_input(
#             "Input search query here:",
#             placeholder="Keywords for most relevant search...",
#             value="gleevac, ascorbic acid",
#         )

#     with c2:
#         max_results = st.number_input("Max papers", value=max_results_current)
#         max_results_current = max_results_current
#     st.markdown("Pre-print server")
#     checks = st.columns(4)
#     with checks[0]:
#         ArXiv_check = st.checkbox("arXiv")
#     with checks[1]:
#         ChemArXiv_check = st.checkbox("chemRxiv")
#     with checks[2]:
#         BioArXiv_check = st.checkbox("bioRxiv")
#     with checks[3]:
#         MedrXiv_check = st.checkbox("medRxiv")

#     searchButton = st.form_submit_button(label="Search")

# if searchButton:
#     # checking which pre-print servers selected
#     XRxiv_servers = []
#     if ArXiv_check:
#         XRxiv_servers.append("rxiv")
#     if ChemArXiv_check:
#         XRxiv_servers.append("chemrxiv")
#     if BioArXiv_check:
#         XRxiv_servers.append("biorxiv")
#     if MedrXiv_check:
#         XRxiv_servers.append("medrxiv")
#     global pdf_info
#     pdf_info = search_click_callback(
#         search_query, max_results, XRxiv_servers=XRxiv_servers
#     )
#     if "pdf_info" not in st.session_state:
#         st.session_state.key = "pdf_info"
#     st.session_state["pdf_info"] = pdf_info


# def answer_callback(question_query, word_count):
#     from langchain import OpenAI

#     llm = OpenAI(temperature=0.7)
#     from langchain.agents import initialize_agent

#     tools = [query2smiles, LLM_predict, searchPapersQA]
#     from gpt_index import GPTListIndex, GPTIndexMemory

#     # index = GPTListIndex([])
#     memory = GPTIndexMemory(
#         index=GPTListIndex([]),
#         memory_key="chat_history",
#         query_kwargs={"response_mode": "compact"},
#     )
#     agent_chain = initialize_agent(
#         tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory
#     )
#     agent_chain.run(input=question_query)
#     st.success("Voila! 😃")


#     import paperqa
#     global docs
#     if docs is None:
#         pdf_info = st.session_state['pdf_info']
#         docs = paperqa.Docs()
#         pdf_paths = [f"{p[4]}/{p[0].replace(':','').replace('/','').replace('.','')}.pdf" for p in pdf_info]
#         pdf_citations = [p[5] for p in pdf_info]
#         print(list(zip(pdf_paths, pdf_citations)))
#         for d, c in zip(pdf_paths, pdf_citations):
#             docs.add(d, c)
#     docs._build_faiss_index()
#     answer = docs.query(question_query,  length_prompt=f'use {word_count:d} words')
#     st.success('Voila! 😃')
#     return answer.formatted_answer

# with st.form(key="question_form", clear_on_submit=False):
#     c1, c2 = st.columns([6, 2])
#     with c1:
#         question_query = st.text_input(
#             "What do you wanna know from these papers?",
#             placeholder="Input questions here...",
#             value="Which molecule is more soluble in water: ascorbic acid or gleevac? show reference and explain why",
#         )
#     with c2:
#         word_count = st.slider(
#             "Suggested number of words in your answer?", 30, 300, 100
#         )
#     submitButton = st.form_submit_button("Submit")

# if submitButton:
#     with st.expander("Found papers:", expanded=True):
#         st.write(f"{st.session_state['all_reference_text']}")
#     with st.spinner("⏳ Please wait..."):
#         start = time.time()
#         final_answer = answer_callback(question_query, word_count)
#         length_answer = len(final_answer)
#         st.text_area("Answer:", final_answer, height=max(length_answer // 4, 100))
#         end = time.time()
#         clock_time = end - start
#         with st.empty():
#             st.write(f"✔️ Task completed in {clock_time:.2f} seconds.")
