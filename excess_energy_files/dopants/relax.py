import os
import glob
import numpy as np
from ase.io import read, write

def add_mass_to_data_file(data_file, poscar_file, num_atom_types=1):
    """
    Adds a mass declaration for all atom types in the LAMMPS data file.
    Inserts the Masses section before the Atoms section.
    
    Parameters:
        data_file (str): Path to the LAMMPS data file.
        num_atom_types (int): Number of atom types in the system (default is 1).
    """
    with open(data_file, 'r') as file:
        lines = file.readlines()

    # Find the position of the "Atoms" section
    atoms_section_index = next((i for i, line in enumerate(lines) if line.strip().startswith("Atoms")), None)
    if atoms_section_index is None:
        raise ValueError("Could not find 'Atoms' section in the LAMMPS data file.")

    # Prepare the Masses section
    masses_section = ["Masses\n\n"]
    for i in range(1, num_atom_types + 1):
        masses_section.append(f"{i} 1.0\n")
    masses_section.append("\n")  # Add a blank line after the Masses section

    # Insert the Masses section before the Atoms section
    lines = lines[:atoms_section_index] + masses_section + lines[atoms_section_index:]

    # Write the updated data file back
    with open(data_file, 'w') as file:
        file.writelines(lines)
    
    # update the potential
    # Read species from POSCAR using ASE
    atoms = read(poscar_file, format='vasp')
    species = sorted(list(set(atoms.get_chemical_symbols())))
    
    # Construct the new pair_coeff line
    pair_coeff_line = f"pair_coeff * * species {' '.join(species)}"
    
    input_file = "relax.in"
    # Read and modify the LAMMPS input file
    with open(input_file, 'r') as f:
        input_lines = f.readlines()
    
    modified_lines = []
    for line in input_lines:
        if line.startswith("pair_coeff"):
            modified_lines.append(pair_coeff_line + "\n")
        else:
            modified_lines.append(line)
    
    # Write the modified file
    with open(input_file, 'w') as f:
        f.writelines(modified_lines)

def run_relaxation_and_extract_energies():
    # Directory containing structure files
    structures_dir = "dopant_cells/"
    
    # Get all files in the "structures" directory, sorted by numeric suffix
    files = sorted(glob.glob(os.path.join(structures_dir, "POSCAR-*")), key=lambda x: int(x.split('-')[-1]))
    
    compounds = ["MoC", "MoH", "MoN"]
    
    from pymatgen.core import Structure
    
    # List to store energies
    energies = []
    for compound in compounds:
        # if the triclinic skew is too large, we need to use the cif file to generate
        # the VASP POSCAR
        if False:
            # Load the CIF file
            structure = Structure.from_file(f"{structures_dir}/{compound}.cif")

            # Save as a POSCAR file
            structure.to(fmt="poscar", filename=f"{structures_dir}/{compound}_POSCAR")
        
        
        file = f"{structures_dir}/{compound}_POSCAR"
        # Copy the current POSCAR file to POSCAR.data
        ase_structure = read(file)*(2,2,2)  # Read using ASE
        write(file, ase_structure, format="vasp",sort=True, direct=True)
        ase_structure = read(file)
        ase_structure.wrap()
        write("POSCAR.data", ase_structure, format="lammps-data")  # Save as LAMMPS data format
        
        # Add per-type mass as 1 in the POSCAR.data file
        add_mass_to_data_file("POSCAR.data", file, len(set(ase_structure.get_chemical_symbols())))

        # Run LAMMPS with the "relax.in" script
        os.system("lmp_serial -in relax.in")

        # Extract energy from "energy.txt"
        try:
            with open("energy.txt", "r") as energy_file:
                energy = float(energy_file.read().strip())  # Assume the file contains just the energy
                energies.append(energy/len(ase_structure))
            np.savetxt(f"{compound}_energy.txt", [energy])
        except Exception as e:
            print(f"Failed to read energy from energy.txt for {file}: {e}")
            energies.append(None)  # Append None if energy extraction fails

    # Save energies to a file
    with open("energies.txt", "w") as energy_output:
        for file, energy in zip(files, energies):
            energy_output.write(f"{file}: {energy}\n")

    print("Energies have been saved to energies.txt")

if __name__ == "__main__":
    run_relaxation_and_extract_energies()
