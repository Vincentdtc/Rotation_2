# Imports
import shutil
import os
import re
from rdkit import Chem
import gzip
from collections import defaultdict
from rdkit.Chem import rdmolops

def extract_sdf_gz_files(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Check if the specific file exists
        for file in files:
            if file.endswith("_docked_vina.sdf.gz"):
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                
                # Define the output file path (remove .gz from the name)
                output_file_path = os.path.join(root, file.replace('.gz', ''))
                
                # Extract the file
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"Extracted: {file_path} -> {output_file_path}")

def process_decoys(sdf_path, number, label, prefix, output_dir, receptor_path, types_file_handle, batch_num=0):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    chembl_counts = defaultdict(int)     # Tracks count per ChEMBL ID
    seen_chembl_ids = set()              # To remember which IDs were seen

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Extract ChEMBL ID from molecule title
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        seen_chembl_ids.add(chembl_id)

        # Only keep up to 8 per ChEMBL ID
        if chembl_counts[chembl_id] >= number:
            continue

        ligand_filename = f'{prefix}_batch{batch_num}_{chembl_id}_{chembl_counts[chembl_id]}.sdf'
        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}) in batch {batch_num}: {e}")
            continue

        unique_code = int(re.findall(r'\d+', chembl_id)[0])
        types_file_handle.write(f'{label} {unique_code} {receptor_path} {ligand_path}\n')
        chembl_counts[chembl_id] += 1

    # Report ChEMBL IDs with fewer than 8 entries
    for chembl_id in seen_chembl_ids:
        if chembl_counts[chembl_id] < number:
            print(f"{chembl_id}: only {chembl_counts[chembl_id]} molecules found")

def process_actives(sdf_path, number, label, prefix, output_dir, receptor_path, types_file_handle):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    
    chembl_counts = defaultdict(int)     # Tracks count per ChEMBL ID
    seen_chembl_ids = set()              # To remember which IDs were seen

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Extract ChEMBL ID from molecule title
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        seen_chembl_ids.add(chembl_id)

        # Only keep up to 8 per ChEMBL ID
        if chembl_counts[chembl_id] >= number:
            continue

        ligand_filename = f'{prefix}_{chembl_id}_{chembl_counts[chembl_id]}.sdf'
        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}): {e}")
            continue

        unique_code = int(re.findall(r'\d+', chembl_id)[0])
        types_file_handle.write(f'{label} {unique_code} {receptor_path} {ligand_path}\n')
        chembl_counts[chembl_id] += 1

    # Report ChEMBL IDs with fewer than 8 entries
    for chembl_id in seen_chembl_ids:
        if chembl_counts[chembl_id] < number:
            print(f"{chembl_id}: only {chembl_counts[chembl_id]} molecules found")

def fix_tetravalent_nitrogens(mol):
    """
    Finds any N atoms whose explicit valence is higher than allowed,
    and sets their formal charge += 1 to relieve the valence error.
    """
    # make sure properties are current
    mol.UpdatePropertyCache(strict=False)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            ev = atom.GetExplicitValence()
            allowed = atom.GetTotalValence()  # permitted max valence for this atom
            if ev > allowed:
                # turn it into an ammonium-like nitrogen
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
    return mol

def process_sdf(infile, outfile):
    # turn off sanitization on load so we can fix before the valence check
    suppl = Chem.SDMolSupplier(infile, sanitize=False, removeHs=False)
    writer = Chem.SDWriter(outfile)
    for mol in suppl:
        if mol is None:
            # parse error or completely dead mol, skip
            continue
        # first pass sanitize: everything except the valence check
        ops = (rdmolops.SanitizeFlags.SANITIZE_ALL
               ^ rdmolops.SanitizeFlags.SANITIZE_PROPERTIES)
        rdmolops.SanitizeMol(mol, ops)
        # fix any bad nitrogens
        mol = fix_tetravalent_nitrogens(mol)
        # now do full sanitization, including valence
        Chem.SanitizeMol(mol)
        writer.write(mol)
    writer.close()