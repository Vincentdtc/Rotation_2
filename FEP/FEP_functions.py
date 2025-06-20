import os
from rdkit import Chem
import math
import os
import pandas as pd

def process_molecules(sdf_path, label, prefix, output_dir, receptor_path, types_file_handle):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    index_to_title = {}  # Dictionary to store index-to-title mapping

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Extract molecule identifier from the title
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        # Set the filename and path for the ligand
        ligand_filename = f'{prefix}_{chembl_id}_{i}.sdf'
        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}): {e}")
            continue

        # Write to the types file
        types_file_handle.write(f'{label} {i} {receptor_path} {ligand_path}\n')

        # Store mapping
        index_to_title[i] = title
    return index_to_title

def deltaG_to_pKd(deltaG_kcal, temperature=297):
    """
    Convert Gibbs free energy (ΔG, in kcal/mol) to pKd (–log10 of the dissociation constant).
    
    Parameters:
        deltaG_kcal (float): Gibbs free energy change (negative for favorable binding), in kcal/mol.
        temperature (float): Temperature in Kelvin (default is 298.15 K).

    Returns:
        float: pKd (dimensionless).
    """
    R = 1.98720425864083e-3  # kcal/mol·K taken from McNutt and Koes 2022 (as well as T value).
    Kd = math.exp(deltaG_kcal / (R * temperature))
    pKd = -math.log10(Kd)
    return pKd

def load_fep_data(file_path):
    """
    Loads FEP benchmark data from a CSV file and returns a nested dictionary
    mapping protein names to ligand names and their corresponding pKd values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A nested dictionary of the form {protein: {ligand: pKd}}.
    """
    df = pd.read_csv(file_path)
    #df['protein'] = df['group_id'].apply(lambda x: x.split('/')[-1])

    result = {}
    for _, row in df.iterrows():
        protein = row['group_id']
        ligand = row['Ligand name']
        exp_dG = row['Exp. dG (kcal/mol)']
        pred_dG = row['Pred. dG (kcal/mol)']
        
        if protein not in result:
            result[protein] = {}
        
        result[protein][ligand] = {'exp_value': deltaG_to_pKd(exp_dG),
                                   'pred_value': deltaG_to_pKd(pred_dG)}

    return result
