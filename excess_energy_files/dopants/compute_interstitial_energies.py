import os
import numpy as np
from ase import io

# Ensure output directory exists
os.makedirs("interstitial_energies", exist_ok=True)

# List of interstitial elements
interstitials = ['B', 'C', 'H', 'N', 'O']

def get_atom_count(structure_path):
    """Reads an ASE structure and returns the number of atoms, filtering out excluded elements."""
    atoms = io.read(structure_path)
    metals = sum(1 for atom in atoms if atom.symbol not in interstitials)
    ints = sum(1 for atom in atoms if atom.symbol in interstitials)
    return metals, ints

def compute_interstitial_energy(metal, interstitial, compound):
    """Calculates interstitial energy and saves it to a file."""
    try:
        # Read energies
        E_compound = np.loadtxt(f"{compound}_energy.txt")
        E_bulk = np.loadtxt(f"{metal}_energy.txt")
        
        # Read atom counts
        metal_atoms_bulk, _ = get_atom_count(f"dopant_cells/{metal}_POSCAR")
        metal_atoms_compound, interstitial_atoms = get_atom_count(f"dopant_cells/{compound}_POSCAR")
        if interstitial_atoms <= 0:
            raise ValueError(f"No interstitial atoms found in {compound}, check structure files.")

        # Compute interstitial energy
        E_i = (E_compound - metal_atoms_compound * (E_bulk / metal_atoms_bulk)) / interstitial_atoms
        
        # Save result
        output_file = f"interstitial_energies/{interstitial}_energy ({compound}).txt"
        np.savetxt(output_file, [E_i])
        print(f"Saved interstitial energy for {interstitial} in {compound} to {output_file}")

    except Exception as e:
        print(f"Error computing interstitial energy for {interstitial} in {compound}: {e}")

# Loop through interstitials and compute energy for each
compounds = [('Cr', 'B', 'CrB'), ('Cr', 'C', 'CrC'), 
             ('Cr', 'H', 'CrH'), ('Cr', 'N', 'CrN'),
             
             ('Fe', 'B', 'FeB'), ('Fe', 'C', 'FeC'), 
             ('Fe', 'H', 'FeH'), ('Fe', 'N', 'FeN'),
             
             ('Mo', 'B', 'MoB'), ('Mo', 'C', 'MoC'), 
             ('Mo', 'H', 'MoH'), ('Mo', 'N', 'MoN'),
             
             ('Ni', 'B', 'NiB'), ('Ni', 'C', 'NiC'),
             ('Ni', 'H', 'NiH'), ('Ni', 'N', 'NiN'),
             
             ('Ti', 'B', 'TiB2'), ('Ti', 'C', 'TiC'), 
             ('Ti', 'H', 'TiH'), ('Ti', 'N', 'TiN')
             ]
for c in compounds:
    compound_name = c[2]  # Adjust compound naming convention if needed
    compute_interstitial_energy(c[0], c[1], c[2])
