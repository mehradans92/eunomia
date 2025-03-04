import streamlit as st  # Web App
import os
from PIL import Image
from tempfile import NamedTemporaryFile
import time
import json

import nltk
nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

import eunomia
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from eunomia.agents import Eunomia
import utils

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
        "This tool allows you to extract information on materials properties from a scientific paper or input txt using natural language. This AI chemist agent uses OpenAI's GPT models, so you must have your own API key. Each query is about 16k tokens, and is charged by OpenAI."
    )
    st.markdown("Used libraries:\n * [Eunomia](https://github.com/AI4ChemS/Eunomia)")
    st.markdown(
        "⚠️ Note that this is only a demo version of the module. You can probably get much better\
                 results with your own customized prompts using the [CLI](https://github.com/AI4ChemS/Eunomia) version."
    )


# title
st.title("Agent-based Learning of Materials Datasets from Scientific Literature")
image = Image.open("img/TOC.png")

col1, col2, col3 = st.columns([1, 2, 3])

# Using the middle column to display the image
with col1:
    st.image(image, width=800)


def main():
    col1, col2 = st.columns([1, 1])
    with col1:
        material = st.text_input(
            "Material type:",
            placeholder="Peptides, MOFs, Dopants, ...",
            value="",
        )
    with col2:
        property = st.text_input(
            "Property of Interest:",
            placeholder="Water Stability, Solubility, ...",
            value="",
        )

    prompt = st.text_area(
        "Input search prompt:",
        value="",
        help=f"You can use this as an example and update your prompt accordingly: {utils.example_input_prompt}",
        height=150,
    )
    with st.expander("Guidelines (optional):", expanded=False):
        rules = st.text_area(
            "Enter your guidelines here:",
            value="",
            help=f"You can use this as an example and update your guidelines accordingly: {utils.example_chemical_rules}",
            height=150,
        )
    with st.expander("Output formatting rules (optional):", expanded=False):
        output_rules = st.text_area(
            "Enter your rules here:",
            value=f"Your final answer should be in a parsable JSON format.",
            help=f"You can use this as an example and update your desired output format accordingly: {utils.example_output_formatting_rules}",
            height=150,
        )
    # Radio button for user choice between uploading a file and entering text
    input_method = st.radio("Choose input method:", ("Upload file", "Input text"))

    with st.form(
        key="columns_in_form",
        clear_on_submit=False,
    ):
        # Conditional logic to display the file uploader or the text input area based on the radio button selection
        if input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload a file",
                type=["pdf", "md", "csv", "txt"],
            )
        elif input_method == "Input text":
            text_input = st.text_area("Enter your text here:")

        # A button to trigger the processing
        process_button = st.form_submit_button("Extract Information")

        if process_button:
            if input_method == "Upload file" and uploaded_file is not None:
                # Process the uploaded file
                extention = os.path.splitext(uploaded_file.name)[-1]
                with NamedTemporaryFile(delete=False, suffix=extention) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name
                docs_processor = eunomia.LoadDoc(file_name=file_path, encoding="utf8")
                sliced_pages = docs_processor.process(
                    [
                        "references ",
                        "acknowledgement",
                        "acknowledgments",
                        "references\n",
                    ],
                    chunk_size=2000,
                    chunk_overlap=50,
                    chunking_type="fixed-size",
                )
            elif input_method == "Input text" and text_input:
                # Process the input text
                docs_processor = eunomia.LoadDoc(text_input=text_input)
                sliced_pages = docs_processor.process(
                    chunk_size=600, chunk_overlap=50, chunking_type="fixed-size"
                )
            st.write("Input loaded.")

            Embedding_model = "text-embedding-3-large"
            faiss_index = FAISS.from_documents(
                sliced_pages, OpenAIEmbeddings(model=Embedding_model)
            )

            tool_kits = utils.tools_generator(
                material, property, prompt, faiss_index, rules
            )
            agent = Eunomia(tools=tool_kits, model="gpt-4", get_cost=True)

            with st.spinner("⏳ Please wait..."):
                start = time.time()
                agent_prompt = f"Read the document and find the mentioned {material}s and their {property}\
                            Check your answers and if the reasoning is not valid, try to check the document again for better reasons.\
                            "
                print(output_rules)
                if len(output_rules) > 0:
                    agent_prompt += agent_prompt + output_rules
                    print(f"agent prompt: {output_rules}")
                result = agent.run(prompt=agent_prompt)
                # Pretty print the JSON with an indent of 4
                final_answer = json.loads(result)
                pretty_json = json.dumps(final_answer, indent=4)
                length_answer = len(pretty_json)
                st.success("Voila! 😃")
                st.text_area(
                    f"{material}s found:",
                    pretty_json,
                    height=max(length_answer // 4, 300),
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
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/mehradansari">Mehrad Ansari</a></h6>',
            unsafe_allow_html=True,
        )
