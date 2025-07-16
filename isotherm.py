import numpy as np
import pandas as pd
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
plt.rcParams['figure.figsize'] = [5, 3.5]


element_mapping = {
    "06. grain_boundaries/boron": "B",
    "06. grain_boundaries/carbon": "C",
    "06. grain_boundaries/hydrogen": "H",
    "06. grain_boundaries/nitrogen": "N"
}

# Define the color of each element, using the matplotlib ordinal color map
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for c, parent_dir in enumerate(element_mapping.keys()):

    element = element_mapping[parent_dir]
    # Define file paths
    segregation_file = "Compiled_Segregation_Energies.csv"
    excess_energy_file = f"{parent_dir}/excess_energy_results.csv"

    # Read segregation energy data from CSV
    segregation_df = pd.read_csv(segregation_file)
    segregation_data = segregation_df.set_index(['Interstitial Type', 'Content (at%)'])['Segregation Energy (eV/atom)'].to_dict()

    # Read excess energy data from CSV
    excess_energy_df = pd.read_csv(excess_energy_file)
    excess_energy_data = {}
    for element in excess_energy_df['Interstitial Type'].unique():
        element_data = excess_energy_df[excess_energy_df['Interstitial Type'] == element]
        ref_comp_data = {}
        for ref_comp in element_data['Ref. Compound'].unique():
            ref_comp_df = element_data[element_data['Ref. Compound'] == ref_comp]
            ref_comp_data[ref_comp] = ref_comp_df.set_index('Content (at%)')['Excess Energy (meV/atom)'].to_dict()
        excess_energy_data[element] = ref_comp_data

    # Define temperature range
    T_range = np.linspace(300, 900, 1000)  # Temperature in Kelvin
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Function to compute segregation isotherm
    def segregation_isotherm(E_seg, Xi, T):    
        """
        Computes the segregation isotherm for a given segregation energy and temperature.

        Parameters:
            E_seg (float): Segregation energy in eV/atom.
            T (float): Temperature in Kelvin.

        Returns:
            float: Segregation isotherm value.
        """

        atoms = read(f"{parent_dir}/CrNi_{element}{int(Xi*100)}/structures/POSCAR-1")
        symbols = atoms.get_chemical_symbols()
        total_atoms = len(symbols)
        # Calculate the atomic fractions for each species
        atomic_fractions = {symbol: symbols.count(symbol) / total_atoms for symbol in set(symbols)}
        # Calculate the configurational entropy
        S_config = -k_B * sum(fraction * np.log(fraction) for fraction in atomic_fractions.values())

        delta_G = E_seg - (T * S_config)
        return (Xi*np.exp(-delta_G / (k_B * T))) / (1 + Xi*np.exp(-delta_G / (k_B * T))) * 100

    

    # Plot competition isotherm for B with each reference compound versus content for each T
    #plt.figure(figsize=(12, 6))
    content_values = sorted(set(key[1] for key in segregation_data.keys() if key[0] == element))

    temps = [300, 1073]
    linestyles = ['-', '--', ':', '-.']

    # Plot segregation curve as a reference in dashed black
    for T in temps:  # Select specific temperatures to plot
        segregation_values = []
        for content in content_values:
            E_seg = segregation_data[(element, content)]
            segregation_value = segregation_isotherm(E_seg, content/100, T)
            segregation_values.append(segregation_value)
        plt.plot(content_values, segregation_values, linestyle=linestyles[temps.index(T)], color=colors[c], label=f"Segregation at T = {T} K", marker='o', linewidth=0.75)

#plt.xlabel(r"$X^{(i)}$ (at%)")
#plt.ylabel(r"$\tilde{X}$ (at%)")
plt.xticks(content_values+[4])
plt.xticks([1,2,4])
plt.ylim(0,22)
plt.yticks([1,2,4,6,8,10,20])
#plt.legend()
plt.grid(True)
plt.savefig("plots/compiled_gb_isotherms.pdf", dpi=450, bbox_inches='tight')
plt.close()
