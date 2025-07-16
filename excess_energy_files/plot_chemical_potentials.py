import json
import numpy as np
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
plt.rcParams['figure.figsize'] = [4.5, 2.5]

# Load the materials database
with open('materials_database.json', 'r') as file:
    data = json.load(file)

# Extract B chemical potential energies for different reference compounds
for interstitial in ["B", "C", "H", "N"]:
    b_data = []
    seen_formulas = set()  # Store formulas to prevent duplicates
    first_occurrence = {}

    # Collect data for B across different metal hosts and compounds
    for metal, entries in data["entries"].get(interstitial, {}).items():
        for entry in entries:
            if "chemical_potential_interstitial" in entry:
                formula = entry["formula"]
                if metal != "Al":
                    b_data.append((entry["formula"], metal, entry["chemical_potential_interstitial"]))
                    seen_formulas.add(formula)  # Mark as seen
                    if metal not in first_occurrence:
                        first_occurrence[metal] = formula

    # Extract values for plotting
    compounds = [item[0] for item in b_data]  # Reference compound names
    metals = [item[1] for item in b_data]  # Corresponding metal hosts
    mu_values = [-item[2] for item in b_data]  # Chemical potential values

    # Assign unique colors to different metal hosts
    unique_metals = sorted(list(set(metals)))
    metal_colors = {metal: plt.cm.tab10(i) for i, metal in enumerate(unique_metals)}
    bar_colors = [metal_colors[metal] for metal in metals]

    # Plot the bar chart
    plt.figure(figsize=(4, 10))
    bars = plt.barh(compounds, mu_values, color=bar_colors, edgecolor='black')

    # Bold the first occurrence of each metal
    for bar, compound, metal in zip(bars, compounds, metals):
        if compound == first_occurrence.get(metal):
            bar.set_linewidth(2.5)  # Make border thicker
            bar.set_edgecolor("black")

    plt.tight_layout()
    plt.gca().invert_yaxis()    
    plt.xlim(0,10)
    plt.savefig(f"{interstitial}_chemical_potentials.pdf", dpi=450, bbox_inches='tight')
    plt.close()
