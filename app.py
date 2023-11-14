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
        "This tool allows you to extract information on water stability property of metal-organic frameworks from a scientific paper or input txt. This AI chemist agent uses OpenAI's GPT models, so you must have your own API key. Each query is about 16k tokens, which costs about only $0.50 on your own API key, charged by OpenAI."
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

# input_prompt = """You are an expert chemist. This document describes the water stability property of few MOFs.
#             For each MOF mentioned find the following:
#             1. The water stability of the MOF.
#             2. The exact sentences without any changes from the document that justifies your decision.
#               Try to find more than once sentence.
#             This should be "Not provided" if you cannot find the water stability for any MOF you find."""

input_prompt = """You are an expert chemist. This document describes the VEGFR affinity property of few peptide sequences.
            For each peptide sequence mentioned, find the following:
            1. The VEGFR affinity of the peptide sequence.
            2. The exact sentences without any changes from the document that justifies your decision.
              Try to find more than once sentence.
            This should be "Not provided" if you cannot find the  VEGFR affinity for any peptide sequence you find.
"""

# chem_rules = f"""
# There are only 3 options for water stability:
# 1. Stable: No change in properties after exposure to moisture or steam, soaking or boiling in water or
#  an aqueous solution.
# Retaining its porous structure in solution.
# No loss of crystallinity.
# insoluble in water or an aqueous solution.
# Water adsorption isotherm should have a steep uptake.
# Good cycling performance.

# 2. Unstable
# The MOF will decompose or change properties or has a change in its crystal structure after exposure/soak to a
#  humid environment, steam or if it partially dissolves in water.
# Soluble or patrially soluble in water or an aqueous solution.


# 3. Not provided,
# If you don't know or cannot justify 1 or 2
# """
chem_rules = f"""A valid peptide sequence should only contain the 20 essential amino acids:
"A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V"
"""

example_output_formatting_rules = """Your final answer should look like:            {
peptide sequence:  VEGFR affinity
 justification: exact sentence from the context.
}"""
# example_output_formatting_rules = "Your final answer should look like:\
#             MOF name: water stability (Stable, Unstable, Not provided)\
#             justification: exact sentence from the context."


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
    # # Add a checkbox for the chain-of-verification
    # chain_of_verification = st.checkbox(
    #     "Add Chain-of-Verification (CoV)",
    #     help=f"Adds an iterative process of verifying agent's answers. Note that this will increase the running time of your information extraction, but will result better answers.",
    # )

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
                    chunk_size=600, chunk_overlap=50, chunking_type="NLTK"
                )
            st.write("Input loaded.")

            Embedding_model = "text-embedding-ada-002"
            faiss_index = FAISS.from_documents(
                sliced_pages, OpenAIEmbeddings(model=Embedding_model)
            )
            # Dynamic tool names list based on checkbox
            # tool_names = ["read_doc"]
            # if chain_of_verification:
            #     tool_names.extend(["eval_justification", "recheck_justification"])
            # tools = EunomiaTools(
            #     tool_names=tool_names,
            #     vectorstore=faiss_index,
            # ).get_tools()
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

    # with st.form(key="visualize_mof", clear_on_submit=False):
    #     cif_file = st.file_uploader(
    #         "Drag and drop or click to upload a file (CIF only)", type="CIF"
    #     )
    #     # A button to trigger the processing
    #     vis_button = st.form_submit_button("Visualize MOF")
    #     if vis_button:
    #         with NamedTemporaryFile(delete=False, suffix=".cif") as tmp_file:
    #             tmp_file.write(cif_file.getvalue())
    #         cif_file_path = tmp_file.name
    #         utils.visualize(cif_file_path)


if __name__ == "__main__":
    main()
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/therealamoo">Mehrad Ansari</a></h6>',
            unsafe_allow_html=True,
        )
