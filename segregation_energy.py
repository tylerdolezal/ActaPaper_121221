import os
import numpy as np
import glob
import pandas as pd
from ase import io
import matplotlib.pyplot as plt

# Global plot settings
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linewidth'] = 0.15
plt.rcParams['grid.color'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True        # Enable grid for axes
plt.rcParams["font.family"] = "Graphik"
plt.rcParams["font.weight"] = "light"
plt.rcParams['figure.figsize'] = [5.0, 2.5]

# Define directories and their corresponding contents
directories = {
    "01. boron": ["B1", "B2", "B4"],
    "02. carbon": ["C1", "C2", "C4"],
    "03. hydrogen": ["H1", "H2", "H4"],
    "04. nitrogen": ["N1", "N2", "N4"]
}

# Define mapping of directory to dopant name
dopant_mapping = {
    "01. boron": "boron",
    "02. carbon": "carbon",
    "03. hydrogen": "hydrogen",
    "04. nitrogen": "nitrogen"
}

element_mapping = {
    "boron": "B",
    "carbon": "C",
    "hydrogen": "H",
    "nitrogen": "N"
}  

# CSV output file
output_file = "Compiled_Segregation_Energies.csv"

# Function to extract the last energy from an energy file
def read_final_energy(file_path):
    try:
        with open(file_path, "r") as f:
            energies = [float(line.strip()) for line in f if line.strip()]
        return energies[-1] if energies else None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to count the number of atoms in a structure
def get_atom_count(structure_path):
    atoms = io.read(structure_path)
    return len(atoms)

# Function to extract interstitial energy from the reference file
def read_interstitial_energy(file_path):
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading interstitial energy file {file_path}: {e}")
        return None

# Initialize results list
results = []

# Loop through each directory and its contents
for directory, contents in directories.items():
    dopant = dopant_mapping[directory]
    element = element_mapping[dopant]

    for content in contents:
        # Find bulk energy files
        bulk_files = glob.glob(os.path.join(directory, f"CrNi_{content}/data_*/energies"))
        bulk_energies = [read_final_energy(f) for f in bulk_files]
        bulk_avg_energy = np.mean([e for e in bulk_energies if e is not None])

        # Find grain boundary energy files
        gb_files = glob.glob(os.path.join("06. grain_boundaries", dopant, f"CrNi_{content}/data_*/energies"))
        gb_energies = [read_final_energy(f) for f in gb_files]
        gb_avg_energy = np.mean([e for e in gb_energies if e is not None])

        # Find structure files to count atoms
        bulk_structure = glob.glob(os.path.join(directory, f"CrNi_{content}/structures/*"))[0]
        gb_structure = glob.glob(os.path.join("06. grain_boundaries", dopant, f"CrNi_{content}/structures/*"))[0]

        # Find undoped bulk and GB energy files
        undoped_bulk_files = glob.glob(os.path.join(directory, f"CrNi_{element}0/data_*/energies"))
        undoped_bulk_energies = [read_final_energy(f) for f in undoped_bulk_files]
        undoped_bulk_avg_energy = np.mean([e for e in undoped_bulk_energies if e is not None])

        undoped_gb_files = glob.glob(os.path.join("06. grain_boundaries", dopant, f"CrNi_{element}0/data_*/energies"))
        undoped_gb_energies = [read_final_energy(f) for f in undoped_gb_files]
        undoped_gb_avg_energy = np.mean([e for e in undoped_gb_energies if e is not None])

        # Undoped structures
        undoped_bulk_structure = glob.glob(os.path.join(directory, f"CrNi_{element}0/structures/*"))[0]
        undoped_gb_structure = glob.glob(os.path.join("06. grain_boundaries/", dopant, f"CrNi_{element}0/structures/*"))[0]

        N_bulk = get_atom_count(bulk_structure)
        N_gb = get_atom_count(gb_structure)
        N_undoped_bulk = get_atom_count(undoped_bulk_structure)
        N_undoped_gb = get_atom_count(undoped_gb_structure)

        def count_interstitials(atoms):
            return len([atom for atom in atoms if atom.symbol in ["B", "C", "H", "N"]])
        N_i = count_interstitials(io.read(gb_structure))

        # Compute per-atom bulk energy
        bulk_energy_per_atom = bulk_avg_energy / N_bulk
        undoped_bulk_energy_per_atom = undoped_bulk_avg_energy / N_undoped_bulk

        # Compute segregation energy with interstitial term
        E_seg = ((gb_avg_energy/N_gb - bulk_energy_per_atom) - (undoped_gb_avg_energy/N_undoped_gb - undoped_bulk_energy_per_atom))
        # Store result
        content_digit = ''.join(filter(str.isdigit, content))
        results.append([element_mapping[dopant], content_digit, E_seg])

# Convert results to DataFrame and save
df = pd.DataFrame(results, columns=["Interstitial Type", "Content (at%)", "Segregation Energy (eV/atom)"])
df.to_csv(output_file, index=False)

print(f"Segregation energy calculations complete. Results saved to {output_file}.")

# Plot segregation energy vs. content for each element, convert to meV
for element in ["B", "C", "H", "N"]:    
    df_element = df[df["Interstitial Type"] == element]
    plt.plot(df_element["Content (at%)"], df_element["Segregation Energy (eV/atom)"] * 1e3, 
             '--', label=element, marker='o', linewidth=0.75)

#plt.xlabel("Content (at%)")
#plt.ylabel(r"$E_{\rm seg}$ (meV/atom)")
plt.yticks(np.arange(-30, 10, 10))
#plt.legend()
plt.savefig("plots/compiled_segregation_energy.pdf", dpi=450, bbox_inches='tight')


