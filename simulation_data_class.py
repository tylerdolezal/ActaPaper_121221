import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import os
import re
import csv
import json
import numpy as np
import pandas as pd
from ase import Atoms
from math import sqrt
from glob import glob
from ase.io import read, write
from ovito.io import import_file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from ase.neighborlist import NeighborList
from collections import Counter, defaultdict
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline
from ase.neighborlist import NeighborList, natural_cutoffs
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
from ovito.modifiers import PolyhedralTemplateMatchingModifier, GrainSegmentationModifier, SelectTypeModifier, DeleteSelectedModifier

# Global materials data base for excess energy
with open("excess_energy_files/materials_database.json", "r") as f:
        materials_data = json.load(f)

# Global plot settings
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linewidth'] = 0.15
plt.rcParams['grid.color'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True        # Enable grid for axes
plt.rcParams["font.family"] = "Graphik"
plt.rcParams["font.weight"] = "light"
plt.rcParams['figure.figsize'] = [4.5, 2.5]

class SimulationData:
    def __init__(self, file_path):
        """
        Initialize the SimulationData object.

        Parameters:
        - file_path (str): Path to the simulation directory.
        """
        self.file_path = file_path
        base_name = os.path.basename(self.file_path)
        try:
            element = base_name.split('_')[1][0]
        except IndexError:
            element = "CrNi"  # Default to base composition for the pure system
            content = 0
        self.parent_dir = os.path.dirname(self.file_path)
        self.output_path = file_path+"/outputs"
        self.structures = glob(f"{file_path}/structures/POSCAR-*")
        self.structures.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.data_dirs = sorted(
    [d for pattern in ["data_*", "data *"] for d in glob(os.path.join(file_path, pattern))]
)       
        
        self.undoped_structures = glob(f"{self.parent_dir}/CrNi_{element}0/structures/POSCAR-*")
        self.undoped_structures.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.undoped_energy = np.mean([self.read_final_energy(f) for f in glob(f"{self.parent_dir}/CrNi_{element}0/data_*/energies")])
        self.undoped_energy_std = np.std([self.read_final_energy(f) for f in glob(f"{self.parent_dir}/CrNi_{element}0/data_*/energies")])
        self.element = element
        self.rdf_data = {}
        self.distance_data = {}
        self.sro_data = {}
        self.avg_rdf = {}
        self.std_rdf = {}
        self.avg_sro = {}
        self.std_sro = {}
        self.distances_dict = {}
        self.interstitials = ['B', 'C', 'H', 'N', 'O']  # Interstitial species
        self.metals = ['Cr', 'Fe', 'Mo', 'Nb', 'Ni', 'Ti']  # Metal species
        
        # Extract the compounds list dynamically using the first entry per metal
        compounds = {}
        chosen_compounds = {}
        for interstitial, metals in materials_data["entries"].items():
            compounds[interstitial] = []
            chosen_compounds[interstitial] = []
            for metal, entries in metals.items():
                if entries and metal in self.metals:  # Check if entries list is not empty
                    # Use the first entry for each metal
                    first_entry = entries[0]
                    compounds[interstitial].append(first_entry["formula"])
                    chosen_compounds[interstitial].append(first_entry)
        self.compounds = compounds

        with open("chosen_compounds.json", "w") as json_file:
            json.dump(chosen_compounds, json_file, indent=4)

        os.makedirs(self.output_path, exist_ok=True)

    def read_final_energy(self, file_path):
        """Reads the last energy entry from an energy file."""
        try:
            with open(file_path, 'r') as f:
                energies = [float(line.strip()) for line in f if line.strip()]
            return energies[-1] if energies else None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def get_atom_counts(self, structure_path):
        """Loads an ASE structure and returns the number of atoms."""
        atoms = read(structure_path)
        metals = sum(1 for atom in atoms if atom.symbol not in self.interstitials)
        ints = sum(1 for atom in atoms if atom.symbol in self.interstitials)
        return metals, ints

    def extract_element_and_content(self, subdir):
        """Extracts the element and content percentage from directory names."""
        base_name = os.path.basename(subdir)
        try:
            element = base_name.split('_')[1][0]  # Extract the element
            content = int(''.join(filter(str.isdigit, base_name)))  # Extract content as integer

        except IndexError:
            element = "CrNi"  # Default to base composition for the pure system
            content = 0
        return element, content
    
    def chemical_potential_energy(self, interstitial, compound):
        """Retrieve the chemical potential of the interstitial species (mu_i) for a given interstitial and compound."""
        # Ensure the interstitial exists in the data
        if interstitial not in materials_data["entries"]:
            return None  # Return None if interstitial is not found

        # Search for the compound within the interstitial category
        for metal, entries in materials_data["entries"][interstitial].items():
            for entry in entries:
                if entry["formula"] == compound:
                    return entry["chemical_potential_interstitial"]  # Return mu_i

        return None  # Return None if compound is not found


        

    def compute_excess_energy(self):
        """Computes excess energy and generates a heatmap."""
        results = []

        undoped_energy_avg = self.undoped_energy
        sigma_undoped = self.undoped_energy_std
        undoped_atoms, _ = self.get_atom_counts(self.undoped_structures[0])

        subdirs = sorted(glob(os.path.join(self.parent_dir, "CrNi_*")))
        element, _ = self.extract_element_and_content(subdirs[0])
        for compound in self.compounds[element]:
            for subdir in subdirs:
                element , interstitial_content = self.extract_element_and_content(subdir)
                print(interstitial_content)
                if interstitial_content > 0:
                    doped_energy_files = glob(f"{subdir}/data_*/energies")
                    doped_energies = [self.read_final_energy(f) for f in doped_energy_files]
                    doped_energy_avg = np.mean([e for e in doped_energies if e is not None])
                    sigma_doped = np.std([e for e in doped_energies if e is not None])
                    doped_structures = sorted(glob(f"{subdir}/structures/*"))
                    metal_atoms, interstitial_atoms = self.get_atom_counts(doped_structures[0])
                    interstitial_energy_ref = self.chemical_potential_energy(element, compound)
                    print(element, compound, interstitial_content, interstitial_energy_ref)
                    
                    excess_energy = (doped_energy_avg - metal_atoms * (undoped_energy_avg / undoped_atoms) - (interstitial_atoms * interstitial_energy_ref)) / (interstitial_atoms + metal_atoms)
                    
                    excess_energy *= 1e3

                    sigma_excess = (1 / (interstitial_atoms + metal_atoms)) * np.sqrt(
                    sigma_doped**2 + (metal_atoms * (sigma_undoped / undoped_atoms))**2
                    )
                    sigma_excess *= 1e3  # Convert to meV

                    # Reduce compound from its true form into Metal+Interstitial form
                    reduced_compound = re.sub(r'\d+', '', compound)
                    # for metal in self.metals also inreduced_compound, make sure it comes before interstitial
                    reduced_compound = f"{[metal for metal in self.metals if metal in reduced_compound][0]}{[interstitial for interstitial in self.interstitials if interstitial in reduced_compound][0]}"

                    results.append([element, reduced_compound, interstitial_content, excess_energy, sigma_excess])

        df = pd.DataFrame(results, columns=["Interstitial Type", "Ref. Compound", "Content (at%)", "Excess Energy (meV/atom)", "Sigma Excess (meV/atom)"])
        # sort dataframe by content and ref. compound
        df = df.sort_values(by=["Content (at%)", "Ref. Compound"])
        df.to_csv(os.path.join(self.parent_dir, "excess_energy_results.csv"), index=False)

        # Filter data for Content (at%) between 1 and 4 for zoomed in look
        # Plot bar chart with error bars
        for filter in [False, True]:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            plt.ylim(-300,600) # for the full plot
            output_file = f"{self.parent_dir}/excess_energy_v_reference_structures.pdf"
            if filter:
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                df = df[(df["Content (at%)"] >= 1) & (df["Content (at%)"] <= 6)]
                plt.ylim(-100,120) # for the magnified plot
                output_file = f"{self.parent_dir}/magnified_excess_energy.pdf"
            # Pivot table for bar chart (x-axis: Interstitial Content, groups: Reference Compound)
            pivot_df = df.pivot(index="Content (at%)", columns="Ref. Compound", values="Excess Energy (meV/atom)")
            # Plot
            #pivot_df.plot(kind="bar", figsize=(12, 6), edgecolor="black")
            # Pivot table for grouped bar chart (x-axis: Interstitial Content, groups: Reference Compound)
            pivot_df = df.pivot(index="Content (at%)", columns="Ref. Compound", values="Excess Energy (meV/atom)")
            pivot_err = df.pivot(index="Content (at%)", columns="Ref. Compound", values="Sigma Excess (meV/atom)")

            

            # Define bar width and positions
            bar_width = 0.15
            num_bars = len(pivot_df.columns)  # Number of bars per group
            x_vals = np.arange(len(pivot_df.index))  # Positions for x-axis

            # Shift bars to center them properly
            offset = (num_bars - 1) * bar_width / 2  # Centering offset

            # Plot bars with error bars
            for i, compound in enumerate(pivot_df.columns):
                if compound != "gas":
                    ax.bar(x_vals - offset + i * bar_width, pivot_df[compound], width=bar_width, label=compound,
                    yerr=pivot_err[compound], capsize=3, alpha=0.8, edgecolor="black")

            # Set labels
            unique_at_percent = sorted(df["Content (at%)"].unique())
            plt.xticks(ticks=np.arange(len(unique_at_percent)), labels=unique_at_percent)
            plt.xticks(rotation=0)
            #plt.legend()
            plt.tight_layout()
            #plt.show()
            plt.savefig(output_file, dpi=350, bbox_inches="tight")
            plt.close()

    def parse_mc_statistics(self, filename):
        """
        Parse the MC statistics from the given file.

        Parameters:
        - filename (str): Path to the MC statistics file.

        Returns:
        - tuple: Acceptance count, rejection count, and steps completed.
        """
        stats = {}
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    key = key.strip()
                    value = value.strip()
                    if key in [
                        'Steps Completed', 'Accepted Swaps', 'New Hosts Accepted',
                        'Translates Accepted', 'Cluster Hops Accepted', 'Cluster Shuffles Accepted',
                        'MD Simulations Accepted', 'Steps for MD'
                    ]:
                        stats[key] = int(value)
                    elif key in ['Acceptance %', 'Rejection %']:
                        stats[key] = float(value)

        steps_completed = stats.get('Steps Completed', 0)
        acceptance_count = stats.get('Acceptance %', 0.0)
        rejection_count = stats.get('Rejection %', 0.0)

        return acceptance_count, rejection_count, steps_completed

    def read_json_file(self, filename):
        """
        Read JSON data from a file.

        Parameters:
        - filename (str): Path to the JSON file.

        Returns:
        - dict: Parsed JSON data.
        """
        with open(filename, 'r') as file:
            return json.load(file)

    def weighted_acceptance(self, data_dir, data, total_steps):
        """
        Calculate weighted acceptance rates for each species.

        Parameters:
        - data (dict): Dictionary of acceptance rates for each species.
        - total_steps (int): Total MC steps completed.

        Returns:
        - dict: Weighted acceptance rates for each species.
        """
        poscar_path = os.path.join(data_dir, "../POSCAR-1")

        cell = read(poscar_path, format="vasp")

        # Get a list of all atomic symbols in the Atoms object
        atomic_symbols = cell.get_chemical_symbols()

        # Count the occurrences of each element
        element_counts = Counter(atomic_symbols)

        # Calculate the total number of atoms
        total_atoms = len(atomic_symbols)

        # Calculate the atomic percent for each species and store in a dictionary
        atomic_percents = {element: count / total_atoms for element, count in element_counts.items()}
        # Species in the cell
        species = set(atomic_symbols)

        for x in species:
            #data[x] = data[x] / (total_steps * atomic_percents[x])
            data[x] = data[x] / (total_steps) * 100

        return data

    def save_weighted_acceptance_rates(self, acceptance_rates_per_system, accept_and_reject):
        """
        Save the weighted acceptance rates and related statistics to a CSV file.

        Parameters:
        - acceptance_rates_per_system (list): List of dictionaries containing acceptance rates for each system.
        - accept_and_reject (list): List of tuples (acceptance rate, rejection rate) for each system.

        Saves:
        - Averages and standard deviations for each species.
        - Total acceptance and rejection rates.
        """
        species_totals = defaultdict(float)
        species_counts = defaultdict(int)
        species_squares = defaultdict(float)

        # Accumulate totals and counts
        for system_rates in acceptance_rates_per_system:
            for species, rate in system_rates.items():
                species_totals[species] += rate
                species_counts[species] += 1

        # Compute averages
        average_acceptance_rates = {
            species: species_totals[species] / species_counts[species]
            for species in species_totals
        }

        # Compute squared deviations
        for system_rates in acceptance_rates_per_system:
            for species, rate in system_rates.items():
                mean = average_acceptance_rates[species]
                species_squares[species] += (rate - mean) ** 2

        # Compute standard deviations
        std_dev_acceptance_rates = {
            species: sqrt(species_squares[species] / species_counts[species])
            for species in species_squares
        }

        # Save to CSV
        csv_filename = os.path.join(self.file_path, "outputs", "averaged_mc_statistics.csv")
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        with open(csv_filename, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Species", "Mean Acceptance Rate", "Standard Deviation"])
            for species in average_acceptance_rates:
                csv_writer.writerow([
                    species,
                    f"{average_acceptance_rates[species]:.4f}",
                    f"{std_dev_acceptance_rates[species]:.4f}",
                ])

            # Calculate overall acceptance/rejection statistics
            accept_values = [ar[0] for ar in accept_and_reject]
            reject_values = [ar[1] for ar in accept_and_reject]

            def calculate_mean_and_std(values):
                if len(values) > 0:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = sqrt(variance)
                else:
                    mean, std_dev = 0, 0
                return mean, std_dev

            accept_mean, accept_std_dev = calculate_mean_and_std(accept_values)
            reject_mean, reject_std_dev = calculate_mean_and_std(reject_values)

            # Write total acceptance/rejection rates
            csv_writer.writerow([])
            csv_writer.writerow(["Metric", "Mean", "Standard Deviation"])
            csv_writer.writerow(["Total Acceptance Rate", f"{accept_mean:.4f}", f"{accept_std_dev:.4f}"])
            csv_writer.writerow(["Total Rejection Rate", f"{reject_mean:.4f}", f"{reject_std_dev:.4f}"])

        print(f"Averaged MC statistics saved: {csv_filename}")

    def process_mc_statistics(self):
        """
        Process MC statistics across all `data_*` directories.
        """
        acceptance_rates_per_system = []
        accept_and_reject = []

        for data_dir in self.data_dirs:
            # Overall Statistics
            stats_file = os.path.join(data_dir, "MonteCarloStatistics")
            accept, reject, total_steps = self.parse_mc_statistics(stats_file)

            # Species Statistics
            species_file = os.path.join(data_dir, "species_counter.json")
            data = self.read_json_file(species_file)

            weighted_acceptance_rates = self.weighted_acceptance(data_dir, data, total_steps)
            acceptance_rates_per_system.append(weighted_acceptance_rates)
            accept_and_reject.append((accept, reject))

        # Save aggregated results
        self.save_weighted_acceptance_rates(acceptance_rates_per_system, accept_and_reject)

    def energy_per_atom(self):

        E = []
        atoms = read(self.structures[0])
        N = len(atoms)

        for data in self.data_dirs:

            energy = np.loadtxt(data+"/energies")[-1]

            E.append(energy/N)


        np.savetxt(self.output_path+"/average_energy_per_atom", [np.mean(E), np.std(E)])

    def read_excess_energy(self):
        """
        Reads the excess energy and its error from a CSV file and stores it in a nested dictionary.

        Returns:
            dict: Nested dictionary of excess energy values and their errors.
        """
        file_path = os.path.join(self.parent_dir, 'excess_energy_results.csv')
        df = pd.read_csv(file_path)
        excess_energy_dict = {}
        for _, row in df.iterrows():
            ref_compound = row['Ref. Compound']
            if ref_compound == 'gas':
                continue
            content = row['Content (at%)']
            excess_energy = row['Excess Energy (meV/atom)'] / 1000  # Convert from meV to eV
            sigma_excess = row['Sigma Excess (meV/atom)'] / 1000  # Convert from meV to eV
            if ref_compound not in excess_energy_dict:
                excess_energy_dict[ref_compound] = {}
            excess_energy_dict[ref_compound][content] = (excess_energy, sigma_excess)
        
        # Alphabetize by reference compound
        excess_energy_dict = dict(sorted(excess_energy_dict.items()))
        
        return excess_energy_dict

    def mclean_isotherm(self, excess_energy, temperature, x_C):
        """
        Calculates the McLean isotherm using the excess energy and configurational entropy.

        Parameters:
            excess_energy (float): Excess energy value.
            temperature (float): Temperature in Kelvin.
            x_C (float): Amount of solute present.

        Returns:
            float: Theta value.
        """
        k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        # Calculate configurational entropy for an ideal system
        atoms = read(f"{self.parent_dir}/CrNi_{self.element}{int(x_C*100)}/structures/POSCAR-1")
        symbols = atoms.get_chemical_symbols()
        total_atoms = len(symbols)
        # Calculate the atomic fractions for each species
        atomic_fractions = {symbol: symbols.count(symbol) / total_atoms for symbol in set(symbols)}
        # Calculate the configurational entropy
        S_config = -k_B * sum(fraction * np.log(fraction) for fraction in atomic_fractions.values())
        # Calculate the McLean isotherm with the entropy term
        exp_term = np.exp(-(excess_energy - temperature * S_config) / (k_B * temperature))
        X_B = (x_C * exp_term) / (1 + (x_C * (exp_term - 1)))
        
        return min(X_B * 100, x_C * 100)   # Convert to percentage

    def mclean_isotherm_with_error(self, excess_energy, sigma_excess, temperature, x_C):
        """
        Calculates the McLean isotherm using the excess energy and configurational entropy,
        and propagates the error in excess energy.

        Parameters:
            excess_energy (float): Excess energy value.
            sigma_excess (float): Standard deviation of the excess energy.
            temperature (float): Temperature in Kelvin.
            x_C (float): Amount of solute present.

        Returns:
            tuple: Theta value and its propagated error.
        """
        k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        # Calculate configurational entropy for an ideal system
        atoms = read(f"{self.parent_dir}/CrNi_{self.element}{int(x_C*100)}/structures/POSCAR-1")
        symbols = atoms.get_chemical_symbols()
        total_atoms = len(symbols)
        # Calculate the atomic fractions for each species
        atomic_fractions = {symbol: symbols.count(symbol) / total_atoms for symbol in set(symbols)}
        # Calculate the configurational entropy
        S_config = -k_B * sum(fraction * np.log(fraction) for fraction in atomic_fractions.values())
        # Calculate the McLean isotherm with the entropy term
        exp_term = np.exp(-(excess_energy - temperature * S_config) / (k_B * temperature))
        X_B = (x_C * exp_term) / (1 + x_C * (exp_term - 1))
        
        # Propagate the error
        dX_B_dE = -x_C * exp_term * (1 - x_C) / (k_B * temperature * (1 + x_C * (exp_term - 1))**2)
        sigma_X_B = abs(dX_B_dE * sigma_excess)
        
        return min(X_B * 100, x_C * 100), sigma_X_B * 100  # Convert to percentage

    def plot_isotherm(self, temperatures):
        """
        Plots the McLean isotherm as a function of content for all reference compounds over different temperatures.

        Parameters:
            temperatures (list): List of temperatures in Kelvin.
            x_C (float): Amount of solute present.
        """
        excess_energy_dict = self.read_excess_energy()

        plt.figure(figsize=(5, 3.5))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', ':', '-.']
        for temp_idx, temperature in enumerate(temperatures):
            for color_idx, (ref_compound, color) in enumerate(zip(excess_energy_dict.keys(), colors)):
                contents = sorted(excess_energy_dict[ref_compound].keys())
                thetas = []
                errors = []
                for content in contents:
                    x_C = content / 100
                    excess_energy, sigma_excess = excess_energy_dict[ref_compound][content]
                    theta, error = self.mclean_isotherm_with_error(excess_energy, sigma_excess,temperature, x_C)
                    thetas.append(theta)
                    errors.append(error)
                
                plt.errorbar(contents, thetas, yerr=errors, marker='o', linestyle=linestyles[temp_idx], 
                             linewidth=0.75, label=f'{ref_compound} at {temperature}K', capsize=3, 
                             color=color)

        plt.xticks(contents, [f'{content}' for content in contents])
        plt.yticks(contents, [f'{content}' for content in contents])
        plt.ylim(-0.5,22)
        plt.savefig(f"{self.parent_dir}/isotherm_plot.pdf", dpi=350, bbox_inches="tight")


    def grab_epoch_data(self, system):
        directory = f'{system}/epochs/'

        results = []
        # Use glob to find all relevant JSON files
        files = sorted(glob(directory+'species_counter_*.json'), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for current_file in files:
            with open(current_file, 'r') as file:
                current_data = json.load(file)
            step = int(current_file.split('_')[-1].split('.')[0])
            results.append({
                'step': step,
                'counts': current_data
            })


        return results
    
    def generate_epoch(self):
        """
        Generate and plot delta C data for all `data_{digit}` subdirectories.
        """
        for data_dir in self.data_dirs:
            results = self.grab_epoch_data(data_dir)

            # Determine the digit from the directory name
            digit = ''.join(filter(str.isdigit, os.path.basename(data_dir)))

            # Path to the `POSCAR-1` file
            poscar_path = os.path.join(data_dir, "../POSCAR-1")

            if not os.path.exists(poscar_path):
                raise FileNotFoundError(f"POSCAR-1 file not found in {poscar_path}")

            cell = read(poscar_path, format="vasp")
            # Extract atomic symbols and counts
            atomic_symbols = cell.get_chemical_symbols()
            element_counts = Counter(atomic_symbols)
            total_atoms = len(atomic_symbols)

            # Calculate the atomic percent for each species and store in a dictionary
            atomic_percents = {element: count / total_atoms for element, count in element_counts.items()}

            # Extract data for plotting
            steps = [result["step"] for result in results]
            species = list(element_counts.keys())

            # Create a plot for each species
            plt.figure(figsize=(10, 6))
            for specie in species:
                count_values = np.array([result["counts"][specie]/result["step"]/atomic_percents[specie] for result in results])
                plt.plot(np.log10(steps), count_values, label=specie)
            
            # Plot customization
            plt.xlabel("MC Steps")
            plt.ylabel("Counts")
            plt.legend()
            plt.grid(True)
            output_file = os.path.join(self.output_path, f"epoch_data-{digit}.png")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Epoch plot saved: {output_file}")

            # Save the epoch data as a csv file, where each column is a species
            csv_output_path = os.path.join(self.output_path, f"epoch_data-{digit}.csv")
            with open(csv_output_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["step"] + species)
                for result in results:
                    row = [result["step"]]
                    for specie in species:
                        count_value = result["counts"].get(specie, 0) / result["step"] / atomic_percents[specie]
                        row.append(count_value)
                    writer.writerow(row)


    def plot_excess_energy(self, ylim=None):
        """
        Generate and plot excess energy per atom across all relevant subdirectories.
        Traverses directories (e.g., `CrNi/` to `CrNi+B20/`), compiles excess energy data,
        and generates a combined plot.
        """

        energy_data = {}

        # Determine the parent directory
        parent_dir = os.path.dirname(self.file_path)

        # Gather directories containing energy data
        subdirs = sorted(glob(os.path.join(parent_dir, "CrNi*")))
        for subdir in subdirs:
            # Path to average energy file in each subdirectory
            energy_file_path = os.path.join(subdir, "outputs", "average_energy_per_atom")
            if not os.path.exists(energy_file_path):
                print(f"Energy file not found: {energy_file_path}")
                continue

            # Determine the element and content from the directory name
            base_name = os.path.basename(subdir)

            try:
                element = base_name.split('_')[1][0]  # Extract the element
                content = int(''.join(filter(str.isdigit, base_name)))  # Extract content as integer
                if content > 10:
                    content = 12
            except IndexError:
                element = "CrNi"  # Default to base composition for the pure system
                content = 0

            # Load reference energies
            if element != "CrNi":
                Ei = np.loadtxt(parent_dir + f"/{element}_energy.txt")
            else:
                Ei = 0

            #primary_path = os.path.join(parent_dir, f"CrNi_{element}0/outputs/average_energy_per_atom")
            fallback_path = os.path.join(parent_dir, "CrNi/outputs/average_energy_per_atom")

            # Try loading from the primary path
            try:
                E0, std0 = np.loadtxt(primary_path)
            except OSError:
                # If primary path fails, fall back to the secondary path
                try:
                    E0, std0 = np.loadtxt(fallback_path)
                except OSError:
                    raise FileNotFoundError(f"E0 not found in either {primary_path} or {fallback_path}")

            if element in ["H", "O"]:
                Ei /= 2  # Divide Ei for diatomic interstitials

            # Load energy data for the current structure
            avg_energy, std = np.loadtxt(energy_file_path)

            # Read the structure to calculate the number of metals and interstitials
            atoms = read(subdir+"/structures/POSCAR-1")
            metal_count = sum(atom.symbol not in self.interstitials for atom in atoms)
            interstitial_count = sum(atom.symbol in self.interstitials for atom in atoms)
            total_atoms = len(atoms)
            # Calculate the excess energy
            excess_energy = (
                (total_atoms*avg_energy) - (metal_count * E0) - (interstitial_count * Ei)
            ) / total_atoms * 1e3

            # Propagate the error
            excess_energy_error = np.sqrt(
                (std * 1e3) ** 2 +
                ((metal_count / total_atoms) * std0 * 1e3) ** 2
            )

            if element not in energy_data:
                energy_data[element] = {"content": [], "excess_energy": [], "std": [],
                                        "energy_atom": [], "energy_atom_std": []}

            energy_data[element]["content"].append(content)
            energy_data[element]["excess_energy"].append(excess_energy)
            energy_data[element]["std"].append(excess_energy_error)
            energy_data[element]["energy_atom"].append(avg_energy)
            energy_data[element]["energy_atom_std"].append(std)

        # Collect all unique content values
        all_contents = sorted(set(content for data in energy_data.values() for content in data["content"]))

        # Save compiled data to a CSV file
        csv_output_path = os.path.join(parent_dir, "compiled_excess_energy.csv")
        with open(csv_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ["Element"] + [f"{content} at%" for content in all_contents]
            writer.writerow(header)

            # Write data rows
            for element, data in energy_data.items():
                row = [element]
                std_row = [f"{element}_std"]  # Row for standard deviation
                energy_row = ["energy_atom"]
                energy_std_row = ["std_energy"]
                for content in all_contents:
                    # Find the index of the content in the list, if it exists
                    if content in data["content"]:
                        index = data["content"].index(content)
                        excess_energy = data["excess_energy"][index]
                        error = data["std"][index]
                        avg_energy = data["energy_atom"][index]
                        std = data["energy_atom_std"][index]
                    else:
                        excess_energy = ""  # Leave empty if no data for this content
                        error = ""
                    row.append(excess_energy)
                    std_row.append(error)
                    energy_row.append(avg_energy)
                    energy_std_row.append(std)
                writer.writerow(row)
                writer.writerow(std_row)
                writer.writerow([])
                writer.writerow(energy_row)
                writer.writerow(energy_std_row)

        print(f"Compiled excess energy data saved: {csv_output_path}")

        # Plot excess energy per atom
        plt.figure(figsize=(4.5, 2.5))
        plt.rcParams["font.family"] = "Graphik"
        plt.rcParams["font.weight"] = "light"

        for element, data in energy_data.items():
            # Sort data by content for smooth plots
            sorted_data = sorted(zip(data["content"], data["excess_energy"], data["std"]))
            sorted_content, sorted_energy, sorted_std = zip(*sorted_data)

            # Calculate error bars
            errors = [sorted_std, sorted_std]

            # Plot with error bars
            errorbar_plot = plt.errorbar(
                sorted_content, sorted_energy, yerr=errors, fmt="o", markersize=4, label=element, capsize=3, linewidth=0.5
            )

            # Extract the color assigned to the error bar plot
            color = errorbar_plot.lines[0].get_color()

            # Create a smooth line using spline interpolation
            sorted_content = np.array(sorted_content)  # Convert to numpy arrays for compatibility
            sorted_energy = np.array(sorted_energy)
            x_smooth = np.linspace(sorted_content.min(), sorted_content.max(), 150)  # Generate smooth x-values
            spline = make_interp_spline(sorted_content, sorted_energy)  # Create a spline
            y_smooth = spline(x_smooth)  # Interpolate y-values

            # Plot the smooth curve
            plt.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=0.75)

        # Customize the plot
        plt.grid(True, which="both")
        plt.xlabel("Content (at%)")
        plt.ylabel("Excess Energy (meV/atom)")
        plt.legend(loc="best", ncol=1)
        plt.tight_layout()
        xtick_labels = [0, 2, 4, 6, 8, 10, 20]  # Replace 12 with 20
        plt.xticks([x for x in range(0, 14, 2)], xtick_labels)
        plt.tick_params(direction="in", which="both")  # Applies to both major and minor ticks
        plt.xlim(0, 12)

        if ylim:
            plt.ylim(ylim[0], ylim[1])

        # Save the plot
        output_file = os.path.join(parent_dir, "Excess_Energy_vs_Content.png")
        plt.savefig(output_file, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"Excess energy plot saved: {output_file}")

    def plot_all_excess_energy(self):
        # Define the directories to walk through
        dirs = ["01. boron", "02. carbon", "03. hydrogen", "04. nitrogen"]

        # Initialize an empty DataFrame to compile data
        compiled_data = pd.DataFrame()
        energy_atom_data = pd.DataFrame()

        # Traverse specified subdirectories
        for dir_name in dirs:
            if os.path.isdir(dir_name):
                # Check if 'compiled_excess_energy.csv' exists in the current directory
                file_path = os.path.join(dir_name, 'compiled_excess_energy.csv')
                if os.path.isfile(file_path):
                    # Read the data and append to the compiled DataFrame
                    try:
                        data = pd.read_csv(file_path)
                        compiled_data = pd.concat([compiled_data, data], ignore_index=True)

                        # Extract energy_atom and std_energy rows for compilation
                        energy_atom_row = data[data['Element'] == 'energy_atom']
                        std_energy_row = data[data['Element'] == 'std_energy']
                        if not energy_atom_row.empty and not std_energy_row.empty:
                            energy_atom_combined = pd.concat([energy_atom_row, std_energy_row], ignore_index=True)
                            energy_atom_combined['Source'] = dir_name  # Track source directory
                            energy_atom_data = pd.concat([energy_atom_data, energy_atom_combined], ignore_index=True)

                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        if compiled_data.empty:
            print("No data found.")
            return

        # Reshape data for plotting
        try:
            # Separate rows for standard deviation
            std_rows = compiled_data[compiled_data['Element'].str.endswith('_std')].copy()
            std_rows['Element'] = std_rows['Element'].str.replace('_std', '')

            # Extract main data
            main_rows = compiled_data[~compiled_data['Element'].str.endswith('_std')]

            # Melt both datasets
            melted_data = main_rows.melt(id_vars=['Element'], var_name='Atomic_Percentage', value_name='Excess_Energy')
            std_data = std_rows.melt(id_vars=['Element'], var_name='Atomic_Percentage', value_name='Std')

            # Process Atomic_Percentage column
            melted_data['Atomic_Percentage'] = melted_data['Atomic_Percentage'].str.replace(' at%', '').astype(float)
            std_data['Atomic_Percentage'] = std_data['Atomic_Percentage'].str.replace(' at%', '').astype(float)

            # Merge excess energy and standard deviation
            plot_data = pd.merge(melted_data, std_data, on=['Element', 'Atomic_Percentage'], suffixes=('', '_std'))

            # Extract energy_atom and std_energy rows
            energy_atom_row = compiled_data[compiled_data['Element'] == 'energy_atom']
            std_energy_row = compiled_data[compiled_data['Element'] == 'std_energy']

            # Plot data with error bars and splines
            plt.figure(figsize=(4.5,5.5))
            for element, group in plot_data.groupby('Element'):
                x = group['Atomic_Percentage']
                y = group['Excess_Energy']

                yerr = group['Std']

                # Create a spline for smooth lines
                x_smooth = np.linspace(x.min(), x.max(), 500)
                spline = make_interp_spline(x, y, k=3)
                y_smooth = spline(x_smooth)

                errorbar_plot = plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=4, label=element, capsize=3, linewidth=0.5)

                # Extract the color assigned to the error bar plot
                color = errorbar_plot.lines[0].get_color()

                plt.plot(x_smooth, y_smooth, c=color, ls='--', linewidth=0.75)

            #plt.xlabel("Light Interstitial Content (at%)")
            #plt.ylabel(r"$E_{Excess}$ (meV/atom)")
            plt.grid(True, which="both")
            plt.ylim(-80,80)
            plt.yticks([x for x in range(-80,120,40)])
            plt.yticks([x for x in range(-80,80,20)], minor=True)
            xtick_labels = [0, 2, 4, 6, 8, 10, 20]  # Replace 12 with 20
            plt.xticks([x for x in range(0,14,2)], xtick_labels)
            plt.xticks([x for x in range(1,13,2)], minor=True)
            plt.xlim(1,12)
            plt.savefig("Excess_energy.pdf", dpi=400, bbox_inches='tight')

            # Plot energy_atom and std_energy combined across all directories
            if not energy_atom_data.empty:
                energy_atom_data = energy_atom_data.melt(id_vars=['Element', 'Source'], var_name='Atomic_Percentage', value_name='Value')
                energy_atom_data['Atomic_Percentage'] = energy_atom_data['Atomic_Percentage'].str.replace(' at%', '', regex=False).astype(float)

                plt.figure(figsize=(4.5,5.5))
                for source, group in energy_atom_data.groupby('Source'):
                    energy_data = group[group['Element'] == 'energy_atom']
                    std_data = group[group['Element'] == 'std_energy']

                    x = energy_data['Atomic_Percentage']
                    y = energy_data['Value']
                    yerr = std_data['Value']

                    # Create a spline for smooth lines
                    x_smooth = np.linspace(x.min(), x.max(), 500)
                    spline = make_interp_spline(x, y, k=3)
                    y_smooth = spline(x_smooth)

                    errorbar_plot = plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=4, label=element, capsize=3, linewidth=0.5)

                    # Extract the color assigned to the error bar plot
                    color = errorbar_plot.lines[0].get_color()

                    plt.plot(x_smooth, y_smooth, c=color, ls='--', linewidth=0.75)

                #plt.xlabel("Light Interstitial Content (at%)")
                plt.grid(True, which="both")
                plt.yticks(np.arange(-5.5,-4.25,0.25))
                plt.yticks(np.arange(-5.5,-4.125,0.125), minor=True)
                plt.ylim(-5.5,-4.5)
                xtick_labels = [0, 2, 4, 6, 8, 10, 20]  # Replace 12 with 20
                plt.xticks([x for x in range(0,14,2)], xtick_labels)
                plt.xticks([x for x in range(1,13,2)], minor=True)
                plt.xlim(1,12)
                plt.grid(True, which="both")
                plt.savefig("Energy_atom.pdf", dpi=400, bbox_inches="tight")

        except Exception as e:
            print(f"Error processing data for plotting: {e}")





    def plot_structure_types(self):
        """
        Generate and plot compiled structure type data across all relevant subdirectories.
        Traverses directories (e.g., `CrNi/` to `CrNi+B20/`), compiles structure data,
        and generates a combined plot with error bars for average structure type counts.
        """
        # Determine the parent directory
        parent_dir = os.path.dirname(self.file_path)

        # Gather directories containing structure count data
        subdirs = sorted(glob(os.path.join(parent_dir, "CrNi*")))

        # Prepare a dictionary to store average and standard deviation data
        compiled_data = {}
        contents = []

        for subdir in subdirs:
            # Path to structure counts CSV in each subdirectory
            csv_path = os.path.join(subdir, "outputs", "structure_counts.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue

            # Extract content percentage from the directory name
            base_name = os.path.basename(subdir)
            try:
                content = int(''.join(filter(str.isdigit, base_name)))  # Extract content as integer
            except ValueError:
                content = 0  # Default for base composition (e.g., CrNi without additions)

            if content == 20:
                content = 12

            if content == 0:
                continue

            contents.append(content)

            # Read the CSV file
            df = pd.read_csv(csv_path)
            reference_structures = ["FCC", "HCP", "BCC", "OTHER"]
            # Compile average and standard deviation for each structure type
            for _, row in df.iterrows():
                structure_type = row["Structure Type"]
                if structure_type not in compiled_data:
                    compiled_data[structure_type] = {"averages": [], "stds": []}

                # Append average and standard deviation for this content
                compiled_data[structure_type]["averages"].append(row["Average"])
                compiled_data[structure_type]["stds"].append(row["STD"])

            # Ensure all reference structures are included
            for structure_type in reference_structures:
                if structure_type not in compiled_data:
                    compiled_data[structure_type] = {"averages": [], "stds": []}

                # Add 0 if missing in current content
                if len(compiled_data[structure_type]["averages"]) < len(contents):
                    compiled_data[structure_type]["averages"].append(0)
                    compiled_data[structure_type]["stds"].append(0)


        # Sort contents and compiled data by content for consistent plotting
        sorted_indices = sorted(range(len(contents)), key=lambda k: contents[k])
        sorted_content = np.array([contents[i] for i in sorted_indices])

        for structure_type in compiled_data:
            compiled_data[structure_type]["averages"] = [
                compiled_data[structure_type]["averages"][i] for i in sorted_indices
            ]
            compiled_data[structure_type]["stds"] = [
                compiled_data[structure_type]["stds"][i] for i in sorted_indices
            ]

        # Plot each structure type
        plt.figure(figsize=(4.5,2.5))
        for structure_type, data in compiled_data.items():

            errorbar_plot = plt.errorbar(
                sorted_content,
                data["averages"],
                yerr=data["stds"],
                label=structure_type,
                fmt="o",
                capsize=3,
                markersize = 4,
                linewidth=0.5
            )

            # Extract the color assigned to the error bar plot
            color = errorbar_plot.lines[0].get_color()

            x_smooth = np.linspace(sorted_content.min(), sorted_content.max(), 150)  # Generate smooth x-values
            spline = make_interp_spline(sorted_content, data["averages"])  # Create a spline
            y_smooth = spline(x_smooth)  # Interpolate y-values

            # Plot the smooth curve
            plt.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=0.75)

        # Formatting the plot
        xtick_labels = [0, 2, 4, 6, 8, 10, 20]  # Replace 12 with 20
        plt.xticks([x for x in range(0,14,2)], xtick_labels)
        plt.xticks([x for x in range(1,13,2)], minor=True)
        plt.xlim(1,12)
        plt.grid(True, which="both")
        plt.tight_layout()

        # Show the plot
        plt.savefig(parent_dir+"/structure_counts.pdf", dpi=400, bbox_inches='tight')
        print(f"Structure Types Figure saved to {parent_dir+"/structure_counts.pdf"}")

        plt.legend()
        # Get the current axis
        ax = plt.gca()

        # Get the legend handles and labels from the axis
        handles, labels = ax.get_legend_handles_labels()

        # Create a new figure for the legend
        legend_fig = plt.figure()  # Adjust size as needed
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")  # Turn off the axis for the legend figure

        legend_ax.legend(handles, labels, loc="center", frameon=True,
        ncols = len(handles))

        # Save the legend as an image
        #plt.savefig(parent_dir+"/legend.pdf", bbox_inches="tight",
        #dpi=400, transparent=True)
        #plt.close(legend_fig)


    def plot_sro(self, ylim=None):
        """
        Generate and plot compiled SRO data across all relevant subdirectories.
        Traverses directories (e.g., `CrNi/` to `CrNi+B20/`), compiles SRO data,
        and generates a combined plot.
        """
        pair_data = {}

        # Determine the parent directory
        parent_dir = os.path.dirname(self.file_path)

        # Gather directories containing SRO data
        subdirs = sorted(glob(os.path.join(parent_dir, "CrNi*")))
        for subdir in subdirs:
            # Path to RDF values CSV in each subdirectory
            csv_path = os.path.join(subdir, "outputs", "RDF_values.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue

            # Determine the element and content from the directory name
            base_name = os.path.basename(subdir)
            try:
                element = base_name.split('_')[1][0]  # Extract the element
                content = int(''.join(filter(str.isdigit, base_name)))  # Extract content as integer
                if content > 10:
                    content = 12
            except IndexError:
                element = "CrNi"  # Default to base composition for the pure system
                content = 0

            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Find the row where "Species" appears
            species_index = df[df.apply(lambda row: row.astype(str).str.contains("Species").any(), axis=1)].index[0]

            sro_data = df.iloc[species_index + 1:].reset_index(drop=True)  # Data after "Species"
            sro_data.columns = df.iloc[species_index]  # Assign proper column names

            # Extract SRO data for each pair
            for _, row in sro_data.iterrows():
                pair_elements = row["Species"].split('-')
                pair = f"{pair_elements[0]}, {pair_elements[1]}"
                sro_value = float(row["SRO"])
                std = float(row["SRO+std"]) - sro_value

                if pair not in pair_data:
                    pair_data[pair] = {"content": [], "SRO": [], "std": []}

                pair_data[pair]["content"].append(content)
                pair_data[pair]["SRO"].append(sro_value)
                pair_data[pair]["std"].append(std)

        # Plot SRO for all pairs on a single plot
        plt.figure(figsize=(4.5,2.5))
        plt.rcParams["font.family"] = "Graphik"
        plt.rcParams["font.weight"] = "light"

        def pair_sort_key(pair):
            elements = pair.split(", ")
            if elements[0] in self.interstitials:
                return (0, elements[0], elements[1])  # Interstitial-metal pairs come first
            if elements[1] in self.interstitials:
                return (0, elements[1], elements[0])  # Interstitial-metal pairs come first (reversed order)
            return (1, elements[0], elements[1])      # Other pairs come second, sorted alphabetically

        # Sort the pairs in pair_data
        sorted_pairs = sorted(pair_data.keys(), key=pair_sort_key)

        for pair in sorted_pairs:
            data = pair_data[pair]
            if pair != f"{element}, {element}":
                # Sort data by content for smooth plots
                sorted_data = sorted(zip(data["content"], data["SRO"], data["std"]))
                sorted_content, sorted_sro, sorted_std = zip(*sorted_data)
                label = pair.split(', ')
                if label[0] in self.interstitials:
                    label[0] = 'i'
                label = f"{label[0]}-{label[1]}"
                # Calculate error bars
                errors = [sorted_std, sorted_std]

                # Plot with error bars
                errorbar_plot = plt.errorbar(sorted_content, sorted_sro, yerr=errors, fmt="o", markersize = 4, label=label, capsize=3, linewidth=0.5)

                # Extract the color assigned to the error bar plot
                color = errorbar_plot.lines[0].get_color()

                # Create a smooth line using spline interpolation
                sorted_content = np.array(sorted_content)  # Convert to numpy arrays for compatibility
                sorted_sro = np.array(sorted_sro)
                x_smooth = np.linspace(sorted_content.min(), sorted_content.max(), 150)  # Generate smooth x-values
                spline = make_interp_spline(sorted_content, sorted_sro)  # Create a spline
                y_smooth = spline(x_smooth)  # Interpolate y-values

                # Plot the smooth curve
                plt.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=0.75)

        # Customize the plot
        plt.grid(True, which="both")
        #plt.xlabel("Content")
        #plt.ylabel("SRO")
        #plt.legend(loc="best", ncols=3)
        plt.tight_layout()
        xtick_labels = [0, 2, 4, 6, 8, 10, 20]  # Replace 12 with 20
        plt.xticks([x for x in range(0,14,2)], xtick_labels)
        plt.xticks([x for x in range(1,13,2)], minor=True)
        plt.axhline(0, c='k', linewidth=0.5)
        # Set ticks inside
        plt.tick_params(direction='in', which='both')  # Applies to both major and minor ticks

        #plt.xticks([])
        #plt.yticks([])
        plt.xlim(0,12)

        if ylim:
            plt.ylim(ylim[0], ylim[1])

        # Save the plot
        output_file = os.path.join(parent_dir, "SRO_vs_Content_All_Pairs.pdf")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(output_file, dpi=400, bbox_inches='tight')

        output_file = os.path.join(parent_dir, "SRO_vs_Content_All_Pairs.png")
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        print(f"SRO plot saved: {output_file}")

    def plot_gb_sro(self, ylim=None):
        """
        Generate and plot compiled SRO data across all relevant subdirectories.
        Traverses directories (e.g., `CrNi/` to `CrNi+B20/`), compiles SRO data,
        and generates a combined plot.
        """
        pair_data = {}

        # Determine the parent directory
        parent_dir = os.path.dirname(self.file_path)

        # Gather directories containing SRO data
        subdirs = sorted(glob(os.path.join(parent_dir, "CrNi*")))
        for subdir in subdirs:
            # Path to RDF values CSV in each subdirectory
            csv_path = os.path.join(subdir, "outputs", "RDF_values.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                continue

            # Determine the element and content from the directory name
            base_name = os.path.basename(subdir)
            try:
                element = base_name.split('_')[1][0]  # Extract the element
                content = int(''.join(filter(str.isdigit, base_name)))  # Extract content as integer
                if content > 3:
                    content = 3
            except IndexError:
                element = "CrNi"  # Default to base composition for the pure system
                content = 0

            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Find the row where "Species" appears
            species_index = df[df.apply(lambda row: row.astype(str).str.contains("Species").any(), axis=1)].index[0]

            sro_data = df.iloc[species_index + 1:].reset_index(drop=True)  # Data after "Species"
            sro_data.columns = df.iloc[species_index]  # Assign proper column names

            # Extract SRO data for each pair
            for _, row in sro_data.iterrows():
                pair_elements = row["Species"].split('-')
                pair = f"{pair_elements[0]}, {pair_elements[1]}"
                sro_value = float(row["SRO"])
                std = float(row["SRO+std"]) - sro_value

                if pair not in pair_data:
                    pair_data[pair] = {"content": [], "SRO": [], "std": []}

                pair_data[pair]["content"].append(content)
                pair_data[pair]["SRO"].append(sro_value)
                pair_data[pair]["std"].append(std)

        # Plot SRO for all pairs on a single plot
        plt.figure(figsize=(3.5, 2.5))
        plt.rcParams["font.family"] = "Graphik"
        plt.rcParams["font.weight"] = "light"

        def pair_sort_key(pair):
            elements = pair.split(", ")
            if elements[0] in self.interstitials:
                return (0, elements[0], elements[1])  # Interstitial-metal pairs come first
            if elements[1] in self.interstitials:
                return (0, elements[1], elements[0])  # Interstitial-metal pairs come first (reversed order)
            return (1, elements[0], elements[1])      # Other pairs come second, sorted alphabetically

        # Sort the pairs in pair_data
        sorted_pairs = sorted(pair_data.keys(), key=pair_sort_key)

        for pair in sorted_pairs:
            data = pair_data[pair]
            if pair != f"{element}, {element}":
                # Sort data by content for smooth plots
                sorted_data = sorted(zip(data["content"], data["SRO"], data["std"]))
                sorted_content, sorted_sro, sorted_std = zip(*sorted_data)
                label = pair.split(', ')
                if label[0] in self.interstitials:
                    label[0] = 'i'
                label = f"{label[0]}-{label[1]}"
                # Calculate error bars
                errors = [sorted_std, sorted_std]

                # Plot with error bars
                plt.plot(sorted_content, sorted_sro, linestyle="--", marker='o', markersize = 4, label=label,linewidth=0.75)

                # Create a smooth line using spline interpolation
                sorted_content = np.array(sorted_content)  # Convert to numpy arrays for compatibility
                sorted_sro = np.array(sorted_sro)
                #x_smooth = np.linspace(sorted_content.min(), sorted_content.max(), 150)  # Generate smooth x-values
                #spline = make_interp_spline(sorted_content, sorted_sro)  # Create a spline
                #y_smooth = spline(x_smooth)  # Interpolate y-values

                # Plot the smooth curve
                #plt.plot(x_smooth, y_smooth, color=color, linestyle="--", linewidth=0.75)

        # Customize the plot
        plt.grid(True, which="both")
        #plt.xlabel("Content")
        #plt.ylabel("SRO")
        #plt.legend(loc="best", ncols=3)
        plt.tight_layout()
        xtick_labels = [0, 1, 2, 4]  # Replace 3 with 4
        plt.xticks([0, 1, 2, 3], xtick_labels)
        plt.xticks([0.5, 1.5, 2.5, 3.5], minor=True)
        plt.axhline(0, c='k', linewidth=0.5)
        # Set ticks inside
        plt.tick_params(direction='in', which='both')  # Applies to both major and minor ticks

        #plt.xticks([])
        #plt.yticks([])
        plt.xlim(0,3)

        if ylim:
            plt.ylim(ylim[0], ylim[1])

        # Save the plot
        output_file = os.path.join(parent_dir, "SRO_vs_Content_All_Pairs.pdf")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(output_file, dpi=400, bbox_inches='tight')

        output_file = os.path.join(parent_dir, "SRO_vs_Content_All_Pairs.png")
        plt.savefig(output_file, dpi=400, bbox_inches='tight')
        print(f"SRO plot saved: {output_file}")


    def kde_heatmap(self, structure_file, bandwidth=0.25):
        """
        Plot a 2D heatmap of interstitial atom positions based on Kernel Density Estimation.

        Parameters:
        atoms : ase.Atoms
            The ASE Atoms object containing the atomic structure.
        interstitials : list of str
            List of atomic symbols corresponding to interstitial elements (e.g., ['B', 'C', 'N', 'O', 'H']).
        bandwidth : float
            Bandwidth parameter for KDE, controlling the smoothness of the density estimation.

        Returns:
        None
        """
        # Determine the parent directory
        V = []

        for i, structure in enumerate(self.structures):
            supercell = read(structure, format="vasp")

            atoms = supercell*(2,2,2)

            # Get interstitial positions
            interstitial_positions = np.array([atom.position for atom in atoms if atom.symbol in self.interstitials])

            if len(interstitial_positions) == 0:
                return(0)
                #raise ValueError("No interstitial atoms found in the provided Atoms object.")

            # Project to 2D (XY plane for simplicity)
            positions_2d = interstitial_positions[:, :2]  # Use X and Y coordinates

            # Perform Kernel Density Estimation (KDE)
            kde = gaussian_kde(positions_2d.T, bw_method=bandwidth)

            # Create a grid for the heatmap
            x_min, y_min = positions_2d.min(axis=0) - 1
            x_max, y_max = positions_2d.max(axis=0) + 1
            x, y = np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150)
            xx, yy = np.meshgrid(x, y)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            from sklearn.cluster import DBSCAN

            dbscan = DBSCAN(eps=3.0, min_samples=2)
            cluster_labels = dbscan.fit_predict(interstitial_positions)

            # Fraction of atoms in clusters
            clustered_fraction = np.sum(cluster_labels != -1) / len(cluster_labels) * 100


            V.append(clustered_fraction)

            # Plot the heatmap
            plt.figure(figsize=(4, 4))
            plt.imshow(zz, extent=[x_min, x_max, y_min, y_max], origin='lower',
            cmap='jet', aspect='auto', vmin=0.0, vmax=0.002)
            #plt.colorbar(label='Density')
            plt.scatter(positions_2d[:, 0], positions_2d[:, 1], color='k', s=10, label='Interstitial Atoms')

            cell = supercell.get_cell()
            y = cell[1,1]
            x = cell[0,0]

            #plt.xlabel("X Position (Å)")
            #plt.ylabel("Y Position (Å)")
            plt.ylim(2.5,24)
            plt.xlim(2.5,24)
            plt.xticks([2.5, 12, 24], [2.5, 12, 24])
            plt.yticks([12, 24], [12, 24])
            #plt.legend()
            output_file = os.path.join(self.output_path, f"kde_heatmap-{i+1}.pdf")
            plt.savefig(output_file, dpi=400, bbox_inches="tight")
            plt.close()
        #print(f"KDE plot saved: {output_file}")
        print(f"{self.file_path}, Average Effective Volume: {np.mean(V)}")


    def calc_rdf(self, file_path, nbins, r_max=None):
        """
        Calculate the RDF and probabilities for a given structure using OVITO.

        Parameters:
        - file_path (str): Path to the atomic structure file (e.g., POSCAR, LAMMPS dump, etc.).
        - nbins (int): Number of bins for RDF calculation.
        - r_max (float): Maximum cutoff distance for RDF calculation.

        Returns:
        - bin_centers (np.ndarray): Array of bin centers for all pairs.
        - rdf (dict): RDF values for each unique pair.
        - probabilities (dict): Probabilities for each unique pair.
        """
        # Load the structure file into an OVITO pipeline
        pipeline = import_file(file_path)
        atoms = read(file_path)
        symbols = atoms.get_chemical_symbols()
        # Determine cutoff radii
        radii = {
            "B": 2.75, "C": 2.75, "H": 2.75, "N": 2.75,
            "O": 2.75, "Cr": 2.8, "Fe": 2.75,
            "Mo": 2.8, "Nb": 2.8, "Ni": 2.8, "Ti": 2.8, "Zr": 3.2
        }
        if r_max is None:
            cutoffs = [radii[symbol] / 2.0 for symbol in symbols]
            r_max = 2 * max(cutoffs)

        # Set up the Coordination Analysis Modifier
        rdf_modifier = CoordinationAnalysisModifier(cutoff=r_max, number_of_bins=nbins, partial=True)
        pipeline.modifiers.append(rdf_modifier)

        # Compute the pipeline to apply the modifier
        data = pipeline.compute()

        # Access the RDF table
        rdf_table = data.tables['coordination-rdf']

        # Extract bin centers (r values)
        bin_centers = rdf_table.xy()[:,0]

        # Extract the total RDF (first y-component)
        total_rdf = rdf_table.y[:, 0]

        # Extract partial RDFs
        partial_rdfs = {}
        bin_centers_dict = {}
        for component, name in enumerate(rdf_table.y.component_names):
            name = tuple(name.split('-'))
            partial_rdfs[name] = rdf_table.y[:, component]
            bin_centers_dict[name] = bin_centers


        return bin_centers_dict, partial_rdfs

    def calculate_probabilities(self, atoms, r_max=None):
        """
        Calculate P(i|j), the conditional probability of finding atom j as a neighbor of atom i.

        Parameters:
        - atoms (ASE Atoms): The atomic configuration.
        - cutoff (float): Cutoff radius for identifying neighbors.

        Returns:
        - probabilities (dict): Dictionary with pairs as keys (e.g., ('Ni', 'Cr'))
                                and conditional probabilities as values.
        """
        # Get chemical symbols
        symbols = atoms.get_chemical_symbols()
        unique_species = sorted(set(symbols))

        # Determine cutoff radii
        radii = {
            "B": 2.75, "C": 2.75, "H": 2.75, "N": 2.75,
            "O": 2.75, "Cr": 2.8, "Fe": 2.75,
            "Mo": 2.8, "Nb": 2.8, "Ni": 2.8, "Ti": 2.8, "Zr": 3.2
        }
        if r_max is None:
            cutoffs = [radii[symbol] / 2.0 for symbol in symbols]

        else:
            cutoffs = [r_max / 2.0] * len(atoms)

        neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighbor_list.update(atoms)

        # Initialize counters
        total_neighbors_by_species = Counter({species: 0 for species in unique_species})  # Total neighbors of each species
        pair_counts = Counter()  # Counts of each pair (i, j)

        # Iterate through all atoms to count neighbors
        for i in range(len(atoms)):
            neighbors, offsets = neighbor_list.get_neighbors(i)
            # Increment neighbor counts
            total_neighbors_by_species[symbols[i]] += len(neighbors)
            for neighbor in neighbors:
                pair = tuple((symbols[i], symbols[neighbor]))
                pair_counts[pair] += 1

        # Calculate probabilities P(i|j)
        probabilities = {}
        for pair, count in pair_counts.items():
            species_i, species_j = pair
            total_neighbors_i = total_neighbors_by_species[species_i]
            probabilities[pair] = count / total_neighbors_i if total_neighbors_i > 0 else 0

        return probabilities


    def calculate_probabilities_with_filtered_neighbors(self, atoms, r_max=None):
        """
        Calculate P(i|j), filtering neighbors by category for metal-metal, metal-interstitial,
        and interstitial-interstitial probabilities.

        Parameters:
        - atoms (ASE Atoms): The atomic configuration.
        - cutoff (float): Cutoff radius for identifying neighbors.
        - interstitials (set): Set of symbols representing interstitial atoms (e.g., {'B', 'C', 'N'}).

        Returns:
        - probabilities (dict): Dictionary with pairs as keys (e.g., ('Ni', 'Cr'))
                                and probabilities as values.
        - categorized_probabilities (dict): Separate dictionaries for 'metal-metal',
                                            'metal-interstitial', and 'interstitial-interstitial'.
        """
        # Get chemical symbols and classify species
        symbols = atoms.get_chemical_symbols()
        unique_species = sorted(set(symbols))
        metals = [s for s in unique_species if s not in self.interstitials]

        # Determine cutoff radii
        radii = {
            "B": 2.75, "C": 2.75, "H": 2.75, "N": 2.75,
            "O": 2.75, "Cr": 2.8, "Fe": 2.75,
            "Mo": 2.8, "Nb": 2.8, "Ni": 2.8, "Ti": 2.8, "Zr": 3.2
        }
        if r_max is None:
            cutoffs = [radii[symbol] / 2.0 for symbol in symbols]

        else:
            cutoffs = [r_max / 2.0] * len(atoms)


        neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
        neighbor_list.update(atoms)

        # Initialize counters for total neighbors and pairs
        total_neighbors_by_category = {
            "metal-metal": Counter({species: 0 for species in metals}),
            "metal-interstitial": Counter({species: 0 for species in unique_species}),
            "interstitial-interstitial": Counter({species: 0 for species in self.interstitials}),
        }
        pair_counts_by_category = {"metal-metal": Counter(), "metal-interstitial": Counter(), "interstitial-interstitial": Counter()}


        # Iterate through all atoms to count neighbors
        for i in range(len(atoms)):
            neighbors, offsets = neighbor_list.get_neighbors(i)
            central_species = symbols[i]

            for neighbor in neighbors:
                neighbor_species = symbols[neighbor]
                pair = (central_species, neighbor_species)  # Sort for symmetry

                # Categorize and count pairs
                if central_species in metals and neighbor_species in metals:
                    pair_counts_by_category["metal-metal"][pair] += 1
                    total_neighbors_by_category["metal-metal"][central_species] += 1
                elif (central_species in self.interstitials and neighbor_species in metals):
                    pair_counts_by_category["metal-interstitial"][pair] += 1
                    total_neighbors_by_category["metal-interstitial"][central_species] += 1
                elif central_species in self.interstitials and neighbor_species in self.interstitials:
                    pair_counts_by_category["interstitial-interstitial"][pair] += 1
                    total_neighbors_by_category["interstitial-interstitial"][central_species] += 1


        # Calculate probabilities P(i|j)
        probabilities = {}
        categorized_probabilities = {"metal-metal": {}, "metal-interstitial": {}, "interstitial-interstitial": {}}

        for category, pair_counts in pair_counts_by_category.items():
            for pair, count in pair_counts.items():
                species_i, species_j = pair
                total_neighbors_i = total_neighbors_by_category[category][species_i]
                # Calculate probabilities based on normalized neighbors
                probabilities[pair] = count / total_neighbors_i if total_neighbors_i > 0 else 0
                categorized_probabilities[category][pair] = probabilities[pair]

        return probabilities, categorized_probabilities



    def generate_rdf(self, nbins, r_max=None):
        """
        Generate RDF and SRO data.

        Parameters:
        - nbins (int): Number of bins for RDF calculation.
        - r_max (float): Maximum cutoff distance for RDF calculation.
        """
        for idx, structure in enumerate(self.structures):
            supercell = read(structure, format="vasp")
            distances, rdf_for_host = self.calc_rdf(structure, nbins, r_max)
            probs, _ = self.calculate_probabilities_with_filtered_neighbors(supercell)
            #probs = self.calculate_probabilities(supercell)

            # Process RDF and SRO values
            labels = sorted(rdf_for_host.keys())
            labels = sorted(
                        rdf_for_host.keys(),
                        key=lambda x: (
                            not (x[0] in self.interstitials or x[1] in self.interstitials),  # Prioritize interstitial pairs
                            x[0] not in self.interstitials,  # Among interstitials, sort by which atom is interstitial
                            sorted(x)  # Alphabetize the pair
                        )
                    )

            for pairs in labels:

                if pairs not in self.rdf_data:
                    self.rdf_data[pairs] = []
                    self.distance_data[pairs] = []

                self.rdf_data[pairs].extend(rdf_for_host[pairs])
                self.distance_data[pairs].extend(distances[pairs])


            labels = sorted(
                        probs.keys(),
                        key=lambda x: (
                            not (x[0] in self.interstitials or x[1] in self.interstitials),  # Prioritize interstitial pairs
                            x[0] not in self.interstitials,  # Among interstitials, sort by which atom is interstitial
                            sorted(x)  # Alphabetize the pair
                        )
                    )
            for pairs in labels:
                i, j = pairs

                c_i = Counter(supercell.get_chemical_symbols())[i] / len(supercell)
                c_j = Counter(supercell.get_chemical_symbols())[j] / len(supercell)

                if c_j > 0:

                    # Handle missing pairs: If the pair is not in probs, assume P(i|j) = 0
                    P_ij = probs.get(pairs, 0)

                    sro_value = 1 - (P_ij / c_j)

                if pairs not in self.sro_data:
                    self.sro_data[pairs] = []

                self.sro_data[pairs].append(sro_value)



            print(f"Finished analyzing POSCAR-{idx+1}")


        self.compute_statistics()

    def compute_statistics(self):
        """
        Compute the average and standard deviation for RDF and SRO values.
        """
        labels = self.rdf_data.keys()
        for species in labels:
            data = list(zip(self.distance_data[species], self.rdf_data[species]))
            distances = [d for d, _ in data]
            unique_distances = sorted(set(distances))

            avg_values = []
            std_values = []

            for d in unique_distances:
                values_at_d = [v for dist, v in data if dist == d]
                avg_values.append(np.mean(values_at_d))
                std_values.append(np.std(values_at_d))

            self.avg_rdf[species] = np.array(avg_values)
            self.std_rdf[species] = np.array(std_values)
            self.distances_dict[species] = np.array(unique_distances)


        labels = self.sro_data.keys()
        for species in labels:
            self.avg_sro[species] = np.mean(self.sro_data[species])
            self.std_sro[species] = np.std(self.sro_data[species])

    def save_rdf(self):
        """
        Save RDF and SRO data to a CSV file.
        """
        output_file = os.path.join(self.output_path, "RDF_values.csv")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, mode="w", newline="") as f:
            csvwriter = csv.writer(f)

            # Write RDF header
            header = ["Distance"]
            for species in self.avg_rdf.keys():
                header.append(f"Avg RDF {species}")
                header.append(f"+sigma {species}")
                header.append(f"-sigma {species}")
            csvwriter.writerow(header)

            # Write RDF data
            distances = list(self.distances_dict.values())[0]
            for k in range(len(distances)):
                row = [distances[k]]
                for species in self.avg_rdf.keys():
                    row.append(self.avg_rdf[species][k])
                    row.append(self.avg_rdf[species][k] + self.std_rdf[species][k])
                    row.append(self.avg_rdf[species][k] - self.std_rdf[species][k])
                csvwriter.writerow(row)

            # Write SRO data
            processed_pairs = set()  # Track pairs already written
            csvwriter.writerow([])  # Add an empty row for separation
            csvwriter.writerow(["Species", "SRO", "SRO+std", "SRO-std"])
            for species in self.avg_sro.keys():

                # Sort the species pair to ensure symmetry handling
                sorted_species = tuple(sorted(species))
                if sorted_species in processed_pairs:
                    continue  # Skip symmetric pair

                # Add to processed pairs
                processed_pairs.add(sorted_species)
                csvwriter.writerow([
                    f"{species[0]}-{species[1]}",
                    self.avg_sro[species],
                    self.avg_sro[species] + self.std_sro[species],
                    self.avg_sro[species] - self.std_sro[species],
                ])

        print(f"RDF data saved to {output_file}")


    def plot_rdf(self, c):
        """
        Plot the RDF data stored in the instance variables with synchronized colors.
        """
        # Ensure RDF data is present
        if not hasattr(self, "distances_dict") or not hasattr(self, "avg_rdf"):
            print("RDF data is not available.")
            return

        parent_dir = os.path.dirname(self.file_path)
        distances = np.array(list(self.distances_dict.values())[0])

        # Define color map
        cmap = plt.get_cmap("tab10")
        color_cycle = list(cmap.colors)  # Convert to list for indexing

        # Categorize species pairs
        interstitial_pairs = []
        interstitial_nickel_pairs = []
        metal_metal_pairs = []

        for species in self.avg_rdf.keys():
            if species[0] in self.interstitials or species[1] in self.interstitials:
                if "Cr" in species:
                    interstitial_pairs.append(species)  # First priority
                elif "Ni" in species:
                    interstitial_nickel_pairs.append(species)  # Second priority
            else:
                metal_metal_pairs.append(species)  # Third priority

        # Ensure species are plotted in the defined order
        ordered_species = interstitial_pairs + interstitial_nickel_pairs + metal_metal_pairs

        # Assign colors based on order:
        species_color_map = {}
        
        # Assign first two colors to interstitial pairs
        if interstitial_pairs:
            species_color_map[interstitial_pairs[0]] = color_cycle[0]  # Blue for "Cr-i"
        if interstitial_nickel_pairs:
            species_color_map[interstitial_nickel_pairs[0]] = color_cycle[1]  # Orange for "i-Ni"

        # Assign metal-metal pairs starting from green (skip blue and orange)
        metal_colors_start = 2  # Start from green (index 2)
        for i, species in enumerate(metal_metal_pairs):
            species_color_map[species] = color_cycle[metal_colors_start + i]

        # Plot RDF for each species
        plt.figure(figsize=(4.5, 2.5))
        for species in ordered_species:
            if species[0] == species[1] and species[0] in self.interstitials:
                print(f"Skipping plot for {species} pair.")
                continue

            avg_rdf = np.array(self.avg_rdf[species])
            std_rdf = np.array(self.std_rdf[species])

            # Plot with assigned color
            plt.errorbar(
                distances,
                avg_rdf,
                yerr=std_rdf,
                fmt="-o",
                markersize=4,
                label=species,
                capsize=3,
                linewidth=0.5,
                color=species_color_map[species],
            )

        plt.grid(True, which="both", linewidth=0.25)
        plt.ylim(0, 20)
        plt.xlim(1, 5)
        xticks = np.arange(1, 5.5, 0.5)  # Generate tick positions
        xticks = xticks[1:]  # Skip the first tick (1)
        plt.xticks(xticks)
        #plt.legend()
        plt.savefig(f"{parent_dir}/rdf-{c}.png", dpi=400, bbox_inches="tight")




    def structure_types(self):
        """
        Analyze the structure using OVITO's Polyhedral Template Matching (PTM).
        Excludes interstitial atoms to focus on the true GB structure.

        Parameters:
            structure: The structure file.
        """
        # Step 1: Apply the Select Type modifier to select interstitial atoms
        # Assuming interstitial atoms are identified by their type or number

        cumulative_counts = defaultdict(int)
        structure_counts_list = []

        for structure in self.structures:
            pipeline = import_file(structure)
            atoms = read(structure)

            interstitial_types = [x for x in self.interstitials if x in atoms.get_chemical_symbols()]  # This should be a list of atomic numbers or symbols
            select_interstitials = SelectTypeModifier(types=interstitial_types)
            pipeline.modifiers.append(select_interstitials)

            # Step 2: Delete selected interstitial atoms
            delete_interstitials = DeleteSelectedModifier()
            pipeline.modifiers.append(delete_interstitials)

            # Step 3: Apply Polyhedral Template Matching (PTM) to analyze GB structure
            ptm_modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0, output_orientation=True)
            pipeline.modifiers.append(ptm_modifier)

            # Step 4: Compute pipeline
            data = pipeline.compute()

            # Extract the PTM structure information
            particle_positions = data.particles['Position']
            structure_types = data.particles['Structure Type']

            # Step 5: Count occurrences of structure types in GB region
            unique, counts = np.unique(structure_types, return_counts=True)

            # Map structure types to their corresponding names (based on OVITO documentation)
            structure_map = {
                PolyhedralTemplateMatchingModifier.Type.OTHER: "OTHER",
                PolyhedralTemplateMatchingModifier.Type.FCC: "FCC",
                PolyhedralTemplateMatchingModifier.Type.HCP: "HCP",
                PolyhedralTemplateMatchingModifier.Type.BCC: "BCC",
                PolyhedralTemplateMatchingModifier.Type.ICO: "Icosahedral"
            }

            # Count structures
            structure_counts = {structure_map[stype]: count/sum(counts) * 100 for stype, count in zip(unique, counts)}
            structure_counts_list.append(structure_counts)

            # Accumulate counts
            for structure_type, count in structure_counts.items():
                cumulative_counts[structure_type] += count

        # Calculate averages
        output_csv = os.path.join(self.output_path, f"structure_counts.csv")
        num_structures = len(self.structures)
        all_structure_types = cumulative_counts.keys()
        average_counts = {key: cumulative_counts[key] / num_structures for key in all_structure_types}
        std_devs = {
        key: np.std([counts.get(key, 0) for counts in structure_counts_list])
        for key in all_structure_types
        }

        # Save results to a CSV file
        with open(output_csv, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header
            header = ["Structure Type"] + [f"Structure {i+1}" for i in range(num_structures)] + ["Average"] + ["STD"]
            csv_writer.writerow(header)

            # Write structure counts and averages
            for structure_type in cumulative_counts.keys():
                row = [structure_type]
                row.extend([counts.get(structure_type, 0) for counts in structure_counts_list])
                row.append(average_counts[structure_type])
                row.append(std_devs[structure_type])
                csv_writer.writerow(row)

        print(f"Structure counts and averages saved to {output_csv}")


    def analyze_gb_structure(self, structure, atoms_gb):
        """
        Analyze the structure of the grain boundary (GB) region using OVITO's Polyhedral Template Matching (PTM).
        Excludes interstitial atoms to focus on the true GB structure.

        Parameters:
            atoms (ASE Atoms): The complete set of atoms in the simulation cell.
            atoms_gb (ASE Atoms): The subset of atoms in the GB region.
        """
        # Step 1: Apply the Select Type modifier to select interstitial atoms
        # Assuming interstitial atoms are identified by their type or number

        pipeline = import_file(structure)
        atoms = read(structure)

        interstitial_types = [x for x in self.interstitials if x in atoms.get_chemical_symbols()]  # This should be a list of atomic numbers or symbols
        select_interstitials = SelectTypeModifier(types=interstitial_types)
        pipeline.modifiers.append(select_interstitials)

        # Step 2: Delete selected interstitial atoms
        delete_interstitials = DeleteSelectedModifier()
        pipeline.modifiers.append(delete_interstitials)

        # Step 3: Apply Polyhedral Template Matching (PTM) to analyze GB structure
        ptm_modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0, output_orientation=True)
        pipeline.modifiers.append(ptm_modifier)

        # Step 4: Compute pipeline
        data = pipeline.compute()

        # Extract the PTM structure information
        particle_positions = data.particles['Position']
        structure_types = data.particles['Structure Type']
        # Step 4: Match positions with atoms_gb
        gb_positions = atoms_gb.positions
        tolerance = 1e-3  # Tolerance for position matching
        gb_indices = []
        for pos in gb_positions:
            distances = np.linalg.norm(particle_positions - pos, axis=1)
            matched_indices = np.where(distances < tolerance)[0]
            gb_indices.extend(matched_indices)

        gb_indices = np.unique(gb_indices)  # Ensure no duplicate indices

        # Step 5: Filter `structure_types` for GB atoms
        gb_structure_types = structure_types[gb_indices]

        # Step 6: Count occurrences of structure types in GB region
        unique, counts = np.unique(gb_structure_types, return_counts=True)

        # Map structure types to their corresponding names (based on OVITO documentation)
        structure_map = {
            PolyhedralTemplateMatchingModifier.Type.OTHER: "OTHER",
            PolyhedralTemplateMatchingModifier.Type.FCC: "FCC",
            PolyhedralTemplateMatchingModifier.Type.HCP: "HCP",
            PolyhedralTemplateMatchingModifier.Type.BCC: "BCC",
            PolyhedralTemplateMatchingModifier.Type.ICO: "Icosahedral"
        }

        # Count structures
        structure_counts = {structure_map[stype]: count/sum(counts) * 100 for stype, count in zip(unique, counts)}
        # Optionally return the structure counts for further analysis
        return structure_counts


    def create_gb_structures(self):
        # Load your input simulation file

        # Ensure the output directory exists
        os.makedirs(self.output_path+"/gb_data/structures", exist_ok=True)

        cumulative_counts = defaultdict(int)
        structure_counts_list = []

        for structure in self.structures:
            pipeline = import_file(structure)
            atoms = read(structure)
            # Step 1: Apply Polyhedral Template Matching (PTM) modifier
            ptm_modifier = PolyhedralTemplateMatchingModifier(output_orientation=True)
            pipeline.modifiers.append(ptm_modifier)

            # Step 2: Apply Grain Segmentation modifier
            grain_modifier = GrainSegmentationModifier()
            pipeline.modifiers.append(grain_modifier)

            # Step 3: Compute the pipeline to apply the modifiers
            data = pipeline.compute()

            grain_orientations = data.tables['grains']['Orientation']

            # Step 4: Access grain segmentation output
            grain_ids = data.particles['Grain']  # Grain IDs for each particle

            # Step 5: Filter atoms by grain IDs
            grain_boundary_mask1 = (grain_ids == 1)
            grain_boundary_mask2 = (grain_ids == 2)

            # Step 6: Create ASE Atoms objects for grain1 and grain2
            atoms_grain1 = atoms[grain_boundary_mask1]
            atoms_grain2 = atoms[grain_boundary_mask2]

            # Step 7: Define cutoff distance and build neighbor list
            cutoff_distance = 2.75
            cutoffs = [cutoff_distance / 2] * len(atoms)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(atoms)

            # Step 8: Create a list to store boundary atoms
            boundary_indices = set()

            # Iterate through atoms in grain1
            for idx in np.where(grain_boundary_mask1)[0]:  # Get indices of grain1 atoms
                indices, offsets = nl.get_neighbors(idx)
                for neighbor in indices:
                    if grain_ids[neighbor] == 2:  # Check if neighbor is in grain2
                        boundary_indices.add(idx)  # Add grain1 atom to boundary
                        boundary_indices.add(neighbor)  # Add grain2 atom to boundary

            # Extract boundary atom indices and create a new ASE Atoms object
            boundary_indices = list(boundary_indices)
            boundary_positions = atoms.positions[boundary_indices]  # Get positions of boundary atoms
            boundary_atom_types = atoms.get_atomic_numbers()[boundary_indices]  # Get atomic numbers of boundary atoms
            cell = atoms.cell  # Retain the simulation cell


            # Convert positions to fractional coordinates
            boundary_fractions = np.linalg.solve(cell.T, boundary_positions.T).T  # Convert to fractional coordinates

            # Filter indices based on fractional z-coordinates
            filtered_indices = [
                i for i, frac in zip(boundary_indices, boundary_fractions) if 0.20 <= frac[2] <= 0.80
            ]

            # Update positions and atom types for filtered atoms
            filtered_positions = atoms.positions[filtered_indices]
            filtered_atom_types = atoms.get_atomic_numbers()[filtered_indices]


            # Create a new ASE Atoms object for boundary atoms
            atoms_gb = Atoms(
                cell=cell,
                positions=filtered_positions,
                numbers=filtered_atom_types,  # Set atomic numbers
                pbc=True  # Retain periodic boundary conditions
            )

            base_name = os.path.basename(structure)
            digit = ''.join(filter(str.isdigit, base_name))
            
            output_file = f"{self.file_path}/structures/POSCARgb-{digit}"
            write(output_file, atoms_gb, format="vasp", direct=True, sort=True)
            print(f"Extracted GB Sim Cell to {output_file}")

            structure_counts = self.analyze_gb_structure(structure, atoms_gb)
            structure_counts_list.append(structure_counts)

            # Accumulate counts
            for structure_type, count in structure_counts.items():
                cumulative_counts[structure_type] += count

        # Calculate averages
        output_csv = os.path.join(self.output_path, f"gb_data/structure_counts.csv")
        num_structures = len(self.structures)
        all_structure_types = cumulative_counts.keys()
        average_counts = {key: cumulative_counts[key] / num_structures for key in all_structure_types}
        std_devs = {
        key: np.std([counts.get(key, 0) for counts in structure_counts_list])
        for key in all_structure_types
        }

        # Save results to a CSV file
        with open(output_csv, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write header
            header = ["Structure Type"] + [f"Structure {i+1}" for i in range(num_structures)] + ["Average"] + ["STD"]
            csv_writer.writerow(header)

            # Write structure counts and averages
            for structure_type in cumulative_counts.keys():
                row = [structure_type]
                row.extend([counts.get(structure_type, 0) for counts in structure_counts_list])
                row.append(average_counts[structure_type])
                row.append(std_devs[structure_type])
                csv_writer.writerow(row)

        print(f"Structure counts and averages saved to {output_csv}")


    def set_gb_path(self):
        self.output_path = self.output_path+"/gb_data"
        self.structures = glob(f"{self.output_path}/structures/*")
        self.structures.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))


    def gb_location(self):

        # Compare against each subsequent structure
        with open(self.output_path+"/average_location", "w") as file:

            for i, structure in enumerate(self.structures):
                atoms = read(structure)
                cell = atoms.get_cell()

                x, y, z = cell[0,0], cell[1,1], cell [2,2]

                # Compute the average position for the current structure
                current_avg_position = np.mean(atoms.positions, axis=0)

                # Output the results
                file.write(f"POSCAR-{i+1}: avg GB location is [{current_avg_position[0]/x:.3f}, {current_avg_position[1]/y:.3f}, {current_avg_position[2]/z:.3f}], {np.linalg.norm(current_avg_position)/np.linalg.norm(np.array([x,y,z]))}\n")

        file.close()



    def gb_concentration(self):
        concentration = []
        all_metal_percents = []
        all_interstitial_percents = []

        with open(self.output_path+"/gb_concentration", "w") as f:
            for i, structure in enumerate(self.structures):
                atoms = read(structure)

                # Get the chemical symbols for all atoms
                symbols = atoms.get_chemical_symbols()

                # Separate metals and interstitials
                metals = [x for x in symbols if x not in self.interstitials]
                interstitials = [x for x in symbols if x in self.interstitials]

                # Count occurrences of each element
                metal_counts = Counter(metals)
                interstitial_counts = Counter(interstitials)

                # Calculate the total number of atoms
                total_atoms = len(symbols)

                # Calculate the total number of metals
                total_metals = len(metals)

                # Calculate the atomic percent for metals (out of total metals)
                metal_atomic_percents = {element: count / total_metals for element, count in metal_counts.items()}

                # Calculate the atomic percent for interstitials (out of all atoms)
                interstitial_atomic_percents = {element: count / total_atoms for element, count in interstitial_counts.items()}

                # Append to the overall lists for averaging later
                all_metal_percents.append(metal_atomic_percents)
                all_interstitial_percents.append(interstitial_atomic_percents)

                # Write individual structure results to the file
                f.write(f"POSCAR-{i+1}\n")
                f.write("Metals:\n")
                for element, percent in metal_atomic_percents.items():
                    f.write(f"  {element}: {percent:.4f}\n")
                f.write("Interstitials:\n")
                for element, percent in interstitial_atomic_percents.items():
                    f.write(f"  {element}: {percent:.4f}\n")
                f.write("\n")

            # Compute averages and standard deviations
            def compute_stats(percent_list, key):
                values = [struct.get(key, 0.0) for struct in percent_list]
                return np.mean(values), np.std(values)

            f.write("Average and Standard Deviation:\n")
            # Calculate and write stats for metals
            f.write("Metals:\n")
            for element in {key for dic in all_metal_percents for key in dic}:
                metal_values = [d.get(element, 0.0) for d in all_metal_percents]
                avg, std = np.mean(metal_values), np.std(metal_values)
                f.write(f"  {element}: Average = {avg:.4f}, Std Dev = {std:.4f}\n")

            # Calculate and write stats for interstitials
            f.write("Interstitials:\n")
            for element in {key for dic in all_interstitial_percents for key in dic}:
                interstitial_values = [d.get(element, 0.0) for d in all_interstitial_percents]
                avg, std = np.mean(interstitial_values), np.std(interstitial_values)
                f.write(f"  {element}: Average = {avg:.4f}, Std Dev = {std:.4f}\n")


    def gb_bond_lengths(self):
        bond_lengths = []
        all_bond_data = defaultdict(list)

        with open(self.output_path+"/gb_bond_lengths", "w") as f:
            for i, structure in enumerate(self.structures):
                atoms = read(structure)

                # Create a NeighborList
                cutoffs = [2.75 / 2] * len(atoms)  # Adjust cutoff as needed
                nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
                nl.update(atoms)

                # Store bond lengths for each pair
                bond_length_data = defaultdict(list)
                positions = atoms.positions
                symbols = atoms.get_chemical_symbols()

                for idx1 in range(len(atoms)):
                    indices, offsets = nl.get_neighbors(idx1)
                    for idx2, offset in zip(indices, offsets):
                        symbol1 = symbols[idx1]
                        symbol2 = symbols[idx2]

                        # Sort symbols for consistent pair key
                        pair = tuple(sorted((symbol1, symbol2)))

                        # Calculate distance
                        pos1 = positions[idx1]
                        pos2 = positions[idx2] + np.dot(offset, atoms.get_cell())
                        distance = np.linalg.norm(pos1 - pos2)

                        # Store bond length
                        bond_length_data[pair].append(distance)

                # Calculate average bond lengths for this structure
                structure_bond_data = {pair: np.mean(lengths) for pair, lengths in bond_length_data.items()}
                bond_lengths.append(structure_bond_data)

                # Append data to all_bond_data for overall statistics
                for pair, avg_length in structure_bond_data.items():
                    all_bond_data[pair].append(avg_length)

                # Write individual bond length data
                f.write(f"POSCAR-{i+1}\n")
                for pair, avg_length in structure_bond_data.items():
                    f.write(f"  {pair}: {avg_length:.4f} A\n")
                f.write("\n")

            # Write overall statistics
            f.write("Average and Standard Deviation for Bond Lengths:\n")
            for pair, lengths in all_bond_data.items():
                avg = np.mean(lengths)
                std = np.std(lengths)
                f.write(f"{pair}: Average = {avg:.4f} A, Std Dev = {std:.4f} A\n")

    def calculate_interstitial_concentration(self, atoms, atoms_gb):
        """
        Calculate the interstitial concentration within the grain boundary atoms.

        Parameters:
            atoms_gb (ASE Atoms): ASE Atoms object representing boundary atoms.
            interstitial_types (list): List of atomic symbols or numbers representing interstitials.

        Returns:
            float: Interstitial concentration (fraction of total atoms).
        """
        # Get atomic numbers for the atoms in the GB
        atom_types = atoms_gb.get_chemical_symbols()
        full_atoms = atoms.get_chemical_symbols()
        metals = [x for x in set(full_atoms) if x not in self.interstitials]

        total_i = sum(1 for atom in full_atoms if atom in self.interstitials)
        total_m = {}
        for x in metals:
            total_m[x] = sum(1 for atom in full_atoms if atom==x)


        # Count interstitials
        interstitial_count = sum(1 for atom in atom_types if atom in self.interstitials)/ total_i if total_i > 0 else 0

        # Count metals
        metal_counts = {}
        for x in metals:
            metal_counts[x] = sum(1 for atom in atom_types if atom==x)/ total_m[x] if total_m[x] > 0 else 0


        return interstitial_count, metal_counts
    
    def calculate_gb_interstitial_concentration(self, atoms, atoms_gb):
        """
        Calculate the interstitial concentration within the grain boundary atoms.

        Parameters:
            atoms_gb (ASE Atoms): ASE Atoms object representing boundary atoms.
            interstitial_types (list): List of atomic symbols or numbers representing interstitials.

        Returns:
            float: Interstitial concentration (fraction of total atoms).
        """
        # Get atomic numbers for the atoms in the GB
        atom_types = atoms_gb.get_chemical_symbols()
        metals = [x for x in set(atom_types) if x not in self.interstitials]

        total_atoms = len(atom_types)


        # Count interstitials
        interstitial_count = sum(1 for atom in atom_types if atom in self.interstitials)/ total_atoms if total_atoms > 0 else 0

        # Count metals
        metal_counts = {}
        for x in metals:
            metal_counts[x] = sum(1 for atom in atom_types if atom==x)/ total_atoms if total_atoms > 0 else 0


        return interstitial_count, metal_counts

    def gb_concentration_over_time(self):
        # Load your input simulation file

        structures = glob(self.file_path+"/structures_over6k/*")
        structures.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

        concentrations = {}

        for structure in structures:
            pipeline = import_file(structure)
            atoms = read(structure)
            atoms.wrap()

            if not any(symbol in self.interstitials for symbol in atoms.get_chemical_symbols()):
                print(f"No interstitials found in {structure}, skipping.")
                continue  # Skip this structure

            # Step 1: Apply Polyhedral Template Matching (PTM) modifier
            ptm_modifier = PolyhedralTemplateMatchingModifier(output_orientation=True)
            pipeline.modifiers.append(ptm_modifier)

            # Step 2: Apply Grain Segmentation modifier
            grain_modifier = GrainSegmentationModifier()
            pipeline.modifiers.append(grain_modifier)

            # Step 3: Compute the pipeline to apply the modifiers
            data = pipeline.compute()

            grain_orientations = data.tables['grains']['Orientation']

            # Step 4: Access grain segmentation output
            grain_ids = data.particles['Grain']  # Grain IDs for each particle

            # Step 5: Filter atoms by grain IDs
            grain_boundary_mask1 = (grain_ids == 1)
            grain_boundary_mask2 = (grain_ids == 2)

            # Step 6: Create ASE Atoms objects for grain1 and grain2
            atoms_grain1 = atoms[grain_boundary_mask1]
            atoms_grain2 = atoms[grain_boundary_mask2]

            # Step 7: Define cutoff distance and build neighbor list
            cutoff_distance = 2.75
            cutoffs = [cutoff_distance / 2] * len(atoms)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(atoms)

            # Step 8: Create a list to store boundary atoms
            boundary_indices = set()

            # Iterate through atoms in grain1
            for idx in np.where(grain_boundary_mask1)[0]:  # Get indices of grain1 atoms
                indices, offsets = nl.get_neighbors(idx)
                for neighbor in indices:
                    if grain_ids[neighbor] == 2:  # Check if neighbor is in grain2
                        boundary_indices.add(idx)  # Add grain1 atom to boundary
                        boundary_indices.add(neighbor)  # Add grain2 atom to boundary

            # Extract boundary atom indices and create a new ASE Atoms object
            boundary_indices = list(boundary_indices)
            boundary_positions = atoms.positions[boundary_indices]  # Get positions of boundary atoms
            cell = atoms.cell  # Retain the simulation cell


            # Convert positions to fractional coordinates
            boundary_fractions = np.linalg.solve(cell.T, boundary_positions.T).T  # Convert to fractional coordinates

            # Filter indices based on fractional z-coordinates
            filtered_indices = [
                i for i, frac in zip(boundary_indices, boundary_fractions) if 0.01 <= frac[2] <= 0.99
            ]

            # Update positions and atom types for filtered atoms
            filtered_positions = atoms.positions[filtered_indices]
            filtered_atom_types = atoms.get_atomic_numbers()[filtered_indices]


            # Create a new ASE Atoms object for boundary atoms
            atoms_gb = Atoms(
                cell=cell,
                positions=filtered_positions,
                numbers=filtered_atom_types,  # Set atomic numbers
                pbc=True  # Retain periodic boundary conditions
            )

            # Calculate interstitial concentration
            i_concentration, m_concentration = self.calculate_interstitial_concentration(atoms, atoms_gb)
            i_gb_concentration, m_gb_concentration = self.calculate_gb_interstitial_concentration(atoms, atoms_gb)
            # Extract MC step (digit) and store the concentration
            
            base_name = os.path.basename(structure)
            digit = int(''.join(filter(str.isdigit, base_name)))

            # Store concentrations in a structured format
            concentrations[digit] = {
                "Interstitial Concentration": i_concentration,
                "Metal Concentrations": {x: m_concentration[x] for x in sorted(m_concentration.keys())},
                "GB Interstitial Concentration": i_gb_concentration,
                "GB Metal Concentrations": {x: m_gb_concentration[x] for x in sorted(m_gb_concentration.keys())}
            }

            print(f"Concentration calculated for MC step {digit}")

        # Write to CSV
        filename = f"{self.file_path}/outputs/gb_concentration_over_time.csv"
        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Extract sorted metal keys to ensure correct column order
            all_metals = sorted(next(iter(concentrations.values()))["Metal Concentrations"].keys())
            all_gb_metals = sorted(next(iter(concentrations.values()))["GB Metal Concentrations"].keys())

            # Define CSV header
            header = (
                ["MC Step", "Interstitial Concentration"]
                + [f"{x} Concentration" for x in all_metals]
                + ["GB Interstitial Concentration"]
                + [f"GB {x} Concentration" for x in all_gb_metals]
            )
            
            writer.writerow(header)  # Write header

            # Write data rows sorted by MC Step
            for step in sorted(concentrations.keys()):
                row = (
                    [step, concentrations[step]["Interstitial Concentration"]]
                    + [concentrations[step]["Metal Concentrations"][x] for x in all_metals]
                    + [concentrations[step]["GB Interstitial Concentration"]]
                    + [concentrations[step]["GB Metal Concentrations"][x] for x in all_gb_metals]
                )
                writer.writerow(row)

    def energy_over_time(self):
        atoms = read(self.structures[0])
        N = len(atoms)
        for data in self.data_dirs:
            energy = np.loadtxt(data+"/energies") / N

            plt.plot(range(len(energy)), energy)

        plt.show()
