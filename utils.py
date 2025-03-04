import eunomia
from eunomia.agents import Eunomia
from langchain.agents import Tool, tool
import langchain

example_input_prompt = """
            You are an expert chemist. This document describes the {property} property of few {material}s.
            For each {material} mentioned find the following:
            1. The {property} of the {material}.
            2. The exact sentences without any changes from the document that justifies your decision.
              Try to find more than once sentence.
            This should be "Not provided" if you cannot find the {property} for any {material} you find."""


example_chemical_rules = """
There are only 3 options for {property}:
1. Stable: No change in properties after exposure to moisture or steam, soaking or boiling in water or\
 an aqueous solution.\
Retaining its porous structure in solution.\
No loss of crystallinity.\
insoluble in water or an aqueous solution.\
Water adsorption isotherm should have a steep uptake.\
Good cycling performance.\

2. Unstable
The MOF will decompose or change properties or has a change in its crystal structure after exposure/soak to a\
 humid environment, steam or if it partially dissolves in water.\
Soluble or partially soluble in water or an aqueous solution.\


3. Not provided,\
If you don't know or cannot justify 1 or 2.
"""


example_output_formatting_rules = "Your final answer should look like:\
            material name: {property} (Stable, Unstable, Not provided)\
            justification: exact sentence from the context."


def visualize(cif_path):
    from pymatgen.io.cif import CifParser
    from pymatgen.io.xyz import XYZ
    import py3Dmol

    # Replace 'path_to_your_file.cif' with the path to your CIF file
    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]

    # Convert the structure to an XYZ format string
    xyz = XYZ(structure).__str__()

    # Set up the py3Dmol view
    view = py3Dmol.view(width=600, height=400)
    view.addModel(xyz, "xyz")
    view.setStyle({"stick": {"radius": 0.15}, "sphere": {"radius": 0.5}})

    from stmol import showmol

    view.zoomTo()
    showmol(view, height=400, width=600)


def tools_generator(
    material, property, prompt, vector_store, rules=None, chain_of_verification=False
):
    """
    Generate a list of tools to be used in the eunomia pipeline.
    """
    if len(rules) > 0:
        added_rules = f"""\nUse the following rules to determine their {property}:
            {rules}"""
        prompt += added_rules

    @tool
    def eval_justification(justification):
        """
        Use this tool to evalaute the justification, and make sure they are valid for every material found.
        """
        from openai import OpenAI
        client = OpenAI()
        model = "gpt-4o"
        prompt = f"""
                Do the below sentences actually talk about the {property} of the found {material}?
                If not, try to find a better justification for that material in the document.

                "{justification}"

                To do this, you should check on the following rules,
                "{rules}"
                """
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
        return response.choices[0].message["content"]

    @tool
    def recheck_justification(material_name):
        """
        Use this tool to check the document again for better justifications for the property of material_name.
        """
        input_prompt = f"""
            You are an expert chemist. The document describes the {property} property of {material_name}.

            Your final answer should contain the following:
            1. The {property} of the {material}.
            3. The exact sentences without any changes from the document that justifies your decision.
              Try to find more than once sentence.
            This should be "Not provided" if you cannot find information on {property}.
            
            """
        if rules is not None:
            input_prompt += f"\nUse the following rules to determine its {property}:\
            {rules}"
        k = 6
        min_k = 2  # Minimum limit for k
        llm = langchain.OpenAI(temperature=0, model_name="gpt-4")
        result = eunomia.RetrievalQABypassTokenLimit(
            input_prompt,
            vector_store,
            k=k,
            min_k=min_k,
            llm=llm,
            search_type="mmr",
            fetch_k=50,
            chain_type="stuff",
            memory=None,
        )
        return result

    @tool
    def read_doc(prompt):
        """
        This tool extracts information from the document and provides exact justifications from document for each answer.
        Always use this tool to get context. Do not make up answers, or change wordings.
        """
        k = 9
        min_k = 2  # Minimum limit for k
        llm = langchain.OpenAI(temperature=0.1, model_name="gpt-4o")
        result = eunomia.RetrievalQABypassTokenLimit(
            prompt,
            vector_store,
            k=k,
            min_k=min_k,
            llm=llm,
            search_type="mmr",
            fetch_k=50,
            chain_type="stuff",
            memory=None,
        )
        return result

    tools = [read_doc, eval_justification, recheck_justification]
    return tools
