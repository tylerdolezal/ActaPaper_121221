import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Global plot settings
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linewidth'] = 0.15
plt.rcParams['grid.color'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True        # Enable grid for axes
plt.rcParams["font.family"] = "Graphik"
plt.rcParams["font.weight"] = "light"
plt.rcParams['figure.figsize'] = [4.0, 2.0]
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.linestyle'] = '--'

def get_atomic_histogram(poscar_file, delta_z=1.0):
    # Read structure from POSCAR
    atoms = read(poscar_file, format='vasp')
    atoms.wrap()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Get z-coordinates
    z_coords = positions[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    delta_z = (z_max - z_min)/20
    
    # Define bins
    bins = np.arange(z_min, z_max + delta_z, delta_z)
    
    # Create a dictionary to hold counts per bin per element
    element_hist = {element: np.zeros(len(bins) - 1) for element in set(symbols)}
    
    # Populate histogram
    for i, z in enumerate(z_coords):
        element = symbols[i]
        bin_index = np.digitize(z, bins) - 1
        if 0 <= bin_index < len(bins) - 1:
            element_hist[element][bin_index] += 1
    
    # Separate metals and interstitials
    interstitials = ['B', 'C', 'H', 'N']
    metals = [element for element in element_hist if element not in interstitials]
    
    # Normalize metals over total count of metals in each layer
    total_metal_counts_per_bin = np.zeros(len(bins) - 1)
    for metal in metals:
        total_metal_counts_per_bin += element_hist[metal]
    
    for metal in metals:
        with np.errstate(divide='ignore', invalid='ignore'):
            element_hist[metal] = np.divide(element_hist[metal], total_metal_counts_per_bin, where=total_metal_counts_per_bin != 0)
    
    for interstitial in interstitials:
        if interstitial in element_hist:
            with np.errstate(divide='ignore', invalid='ignore'):
                element_hist[interstitial] = np.divide(element_hist[interstitial], total_metal_counts_per_bin, where=total_metal_counts_per_bin != 0)
    
    return bins, element_hist

def plot_histogram(bins, element_hist, system, alpha=1.0, save=True, marker='o'):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    interstitials = ['B', 'C', 'H', 'N']
    not_interstitials = sorted([element for element in element_hist if element not in interstitials])
    
    # follow the matplotlib color wheel cycle for plotting
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create a dictionary for element colors
    element_colors = {element: colors[0] for element in interstitials}
    for i, element in enumerate(not_interstitials):
        element_colors[element] = colors[(i % (len(colors) - 1)) + 1]
    
    # sync colors across systems
    if system == "CrNi_B0":
        plt.plot([],[])

    for i, element in enumerate(interstitials + not_interstitials):
        if element in element_hist:
            plt.plot(bin_centers, element_hist[element] * 100, marker=marker, label=element, color = element_colors[element], alpha=alpha)
            # print the max value and the corresponding bin

            if element in interstitials:
                #mask = ((bin_centers >= 0) & (bin_centers <= 5)) | ((bin_centers >= 18) & (bin_centers <= 25)) | ((bin_centers >= 35) & (bin_centers <= 40))
                max_value = np.max(element_hist[element])*100
                print(f"{element}: {max_value}")

    #plt.xlabel(r'$\hat{z}$ (Å)')
    #plt.ylabel('Atomic Fraction (at%)')
    plt.xlim(0, 38.5)
    plt.ylim(0, 80)
    plt.xticks(np.arange(0, 39.25, 5), minor=True)
    plt.yticks(np.arange(0, 81, 10), minor=True)
    
    if save:
        plt.savefig(f"plots/{system}.pdf", dpi=450, bbox_inches='tight')
    

poscar_files=[
    "06. grain_boundaries/boron/CrNi_B1/POSCAR-1",
    "06. grain_boundaries/boron/CrNi_B2/POSCAR-1",
    "06. grain_boundaries/boron/CrNi_B4/POSCAR-1",
    "06. grain_boundaries/carbon/CrNi_C1/POSCAR-1",
    "06. grain_boundaries/carbon/CrNi_C2/POSCAR-1",
    "06. grain_boundaries/carbon/CrNi_C4/POSCAR-1",
    "06. grain_boundaries/hydrogen/CrNi_H1/POSCAR-1",
    "06. grain_boundaries/hydrogen/CrNi_H2/POSCAR-1",
    "06. grain_boundaries/hydrogen/CrNi_H4/POSCAR-1",
    "06. grain_boundaries/nitrogen/CrNi_N1/POSCAR-1",
    "06. grain_boundaries/nitrogen/CrNi_N2/POSCAR-1",
    "06. grain_boundaries/nitrogen/CrNi_N4/POSCAR-1",
]

grouped_systems = {
    "B": ["CrNi_B1", "CrNi_B2", "CrNi_B4"],
    "C": ["CrNi_C1", "CrNi_C2", "CrNi_C4"],
    "H": ["CrNi_H1", "CrNi_H2", "CrNi_H4"],
    "N": ["CrNi_N1", "CrNi_N2", "CrNi_N4"],
}

dirs = {
    "B": "boron",
    "C": "carbon",
    "H": "hydrogen",
    "N": "nitrogen",
}

# Plot B0 separately
b0_file = "06. grain_boundaries/boron/CrNi_B0/POSCAR-1"
bins, element_hist = get_atomic_histogram(b0_file)
plt.figure()
plot_histogram(bins, element_hist, "CrNi_B0")
plt.close()

# Plot B1, B2, B4 together with different alpha values
markers = ['o', 's', '^']
for interstitial, systems in grouped_systems.items():
    plt.figure()
    for i, system in enumerate(systems):
        poscar_file = f"06. grain_boundaries/{dirs[interstitial]}/" + system + "/POSCAR-1"
        bins, element_hist = get_atomic_histogram(poscar_file)
        plot_histogram(bins, element_hist, system, alpha=0.3 + 0.3 * i, save=False, marker=markers[i])
    
    #plt.xlabel(r'$\hat{z}$ (Å)')
    #plt.ylabel('Atomic Fraction (at%)')
    plt.xlim(0, 38.25)
    plt.ylim(0, 80)
    plt.xticks(np.arange(0, 39.25, 5), minor=True)
    plt.yticks(np.arange(0, 81, 10), minor=True)    
    plt.savefig(f"plots/CrNi_{interstitial}.pdf", dpi=450, bbox_inches='tight')
    plt.close()
