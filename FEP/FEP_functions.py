import os
import math
import pandas as pd
from rdkit import Chem


def process_molecules(sdf_path, label, prefix, output_dir, receptor_path, types_file_handle):
    """
    Processes molecules from an SDF file and writes individual ligand files.

    Args:
        sdf_path (str): Path to input SDF file.
        label (str): Dataset label (e.g., train/test/valid).
        prefix (str): Prefix for output filenames.
        output_dir (str): Directory to store individual ligand SDF files.
        receptor_path (str): Path to receptor file (used in types file).
        types_file_handle (file): Open file handle to write ligand metadata lines.

    Returns:
        dict: Mapping of index to original molecule title.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    index_to_title = {}

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Retrieve molecule name or use fallback
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        # Define output path
        ligand_filename = f'{prefix}_{chembl_id}_{i}.sdf'
        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}): {e}")
            continue

        # Write ligand metadata line to types file
        types_file_handle.write(f'{label} {i} {receptor_path} {ligand_path}\n')

        # Record mapping of index to molecule title
        index_to_title[i] = title

    return index_to_title


def deltaG_to_pKd(deltaG_kcal, temperature=297):
    """
    Convert ΔG (kcal/mol) to pKd using the thermodynamic relationship:
        pKd = -log10(Kd) where Kd = exp(ΔG / (RT))

    Args:
        deltaG_kcal (float): Gibbs free energy in kcal/mol.
        temperature (float): Temperature in Kelvin. Default is 297 K.

    Returns:
        float: pKd value.
    """
    R = 1.98720425864083e-3  # kcal/(mol·K), universal gas constant
    Kd = math.exp(deltaG_kcal / (R * temperature))
    return -math.log10(Kd)


def load_fep_data(file_path):
    """
    Load FEP benchmark data from a CSV file into a nested dictionary format.

    Args:
        file_path (str): Path to the CSV file with FEP data.

    Returns:
        dict: Nested dictionary in the form:
              {
                protein_1: {
                    ligand_1: {'exp_value': pKd1, 'pred_value': pKd2},
                    ...
                },
                ...
              }
    """
    df = pd.read_csv(file_path)

    fep_dict = {}
    for _, row in df.iterrows():
        protein = row['group_id']
        ligand = row['Ligand name']
        exp_dG = row['Exp. dG (kcal/mol)']
        pred_dG = row['Pred. dG (kcal/mol)']

        if protein not in fep_dict:
            fep_dict[protein] = {}

        fep_dict[protein][ligand] = {
            'exp_value': deltaG_to_pKd(exp_dG),
            'pred_value': deltaG_to_pKd(pred_dG)
        }

    return fep_dict
