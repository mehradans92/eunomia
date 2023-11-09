import streamlit as st  # Web App


import os
from PIL import Image
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

api_key_url = (
    "https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key"
)

with st.sidebar:
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

    st.markdown(
        "This tool allows you to extract information on water stability property of metal-organic frameworks from a scientific paper or input txt. This AI chemist agent uses OpenAI's GPT models, so you must have your own API key. Each query is about 16k tokens, which costs about only $0.50 on your own API key, charged by OpenAI."
    )
    st.markdown("Used libraries:\n * [Eunomia](https://github.com/AI4ChemS/Eunomia)")


text_input = ""
uploaded_file = ""


docs = None
api_key = " "

# title
st.title("Agent-based Learning of Materials Datasets from Scientific Literature")
image = Image.open("img/TOC.png")

col1, col2, col3 = st.columns([1, 2, 3])

# Using the middle column to display the image
with col1:
    st.image(image, width=800)


def main():
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
    process_button = st.button("Extract Information")

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
                st.success("Voila! 😃")
                st.text_area(
                    "Answer:", pretty_json, height=max(length_answer // 4, 200)
                )
                end = time.time()
                clock_time = end - start
                with st.empty():
                    st.write(f"✔️ Task completed in {clock_time:.2f} seconds.")


if __name__ == "__main__":
    main()
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/therealamoo">Mehrad Ansari</a></h6>',
            unsafe_allow_html=True,
        )
