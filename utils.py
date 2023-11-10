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
