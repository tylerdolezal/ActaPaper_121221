import os
import csv
import glob
import numpy as np
from ase.io import read
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
    "01. boron": ["B0", "B1", "B2", "B4"],
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
output_file = "Compiled_GB_Energies.csv"
conversion_factor = 16.02176/2
# Write the CSV header
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Interstitial Type", "Content", "E0 (eV/atom)", "Ef (eV/atom)", "eGB (J/m²)"])

# Loop over each directory and its contents
for directory, contents in directories.items():
    dopant = dopant_mapping[directory]  # Get the dopant name
    element = element_mapping[dopant]
    for content in contents:
        # Step 1: Calculate E0 (Bulk Energy Per Atom)
        bulk_energy_files = glob.glob(f"{directory}/CrNi_{content}/data_*/energies")
        bulk_energies = []

        for energy_file in bulk_energy_files:
            try:
                energies = np.loadtxt(energy_file)
                digit = energy_file.split("_")[-1].split("/")[0]  # Extract digit
                atoms = read(f"{directory}/CrNi_{content}/structures/POSCAR-{digit}")
                bulk_energies.append(energies[-1] / len(atoms))  # Normalize energy per atom
            except Exception as e:
                print(f"Error processing {energy_file}: {e}")

        # Average bulk energy per atom
        if bulk_energies:
            E0 = np.mean(bulk_energies)

        else:
            E0 = None  # Handle missing data

        # Step 2: Calculate Ef (Grain Boundary Energy) and A (Area) for each structure
        gb_energy_files = glob.glob(f"06. grain_boundaries/{dopant}/CrNi_{content}/data_*/energies")
        final_energies = []
        e_gb_contributions = []  # Store grain boundary energy contributions

        for energy_file in gb_energy_files:
            try:
                energies = np.loadtxt(energy_file)
                Ef = energies[-1]
                digit = energy_file.split("_")[-1].split("/")[0]  # Extract digit
                atoms = read(f"06. grain_boundaries/{dopant}/CrNi_{content}/structures/POSCAR-{digit}")
                final_energies.append(Ef / len(atoms))
                A = atoms.get_cell()[0, 0] * atoms.get_cell()[1, 1]  # Area in the x-y plane
                e_gb_contrib = (Ef - (len(atoms)*E0)) / (2*A) * (conversion_factor)  # Convert to J/m² (divide by 2 for 2 GBs)
                e_gb_contributions.append(e_gb_contrib)
            except Exception as e:
                print(f"Error processing {energy_file}: {e}")

        # Average grain boundary energy
        if final_energies and e_gb_contributions:
            Ef = np.mean(final_energies)
            e_gb_avg = np.mean(e_gb_contributions)
            print(e_gb_avg, np.std(e_gb_contributions))
        else:
            Ef, e_gb_avg = None, None  # Handle missing data

        content_digit = ''.join(filter(str.isdigit, content))
        # Write the results to the CSV file
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([element, content_digit, f"{E0:.6f}" if E0 else "N/A",
                             f"{Ef:.6f}" if Ef else "N/A", f"{e_gb_avg:.6f}" if e_gb_avg else "N/A"])

print(f"Results saved to {output_file}")

import pandas as pd

# plot GB energy vs. content for each element
df = pd.read_csv(output_file)
df["Content"] = df["Content"].replace(4, 3)  # Replace content 4 with 3
for element in ["B", "C", "H", "N"]:
    df_element = df[(df["Interstitial Type"] == element) & (df["Content"].astype(int) > 0)]
    plt.plot(df_element["Content"], df_element["eGB (J/m²)"], '--', label=element, marker='o', linewidth=0.75)

# Plot horizontal line for "B" at the eGB value for B, 0
eGB_B0 = df[(df["Interstitial Type"] == "B") & (df["Content"].astype(int) == 0)]["eGB (J/m²)"].values[0]
plt.axhline(eGB_B0, color='k', linestyle='-', linewidth=0.75, label=f'B, 0 ({eGB_B0:.2f} J/m²)')

#plt.xlabel("Content (at%)")
#plt.ylabel(r"$\gamma_{\rm GB}$ (J/m²)")
plt.yticks(np.linspace(0.4, 0.9, 6))
xlabels = [1,2,4]
plt.xticks(df["Content"].astype(int).unique()[1:], xlabels)
#plt.legend()
plt.savefig("plots/compiled_gb_energy.pdf", dpi=450, bbox_inches='tight')