import os
import numpy as np
import glob
import pandas as pd
from ase import io
import matplotlib.pyplot as plt

# reach into 06. grain_boundaries folder for each element, go into outputs folder
# and grab the gb_concentration_over_time.csv file, average from step 8500 to the final step

# Global plot settings
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linewidth'] = 0.15
plt.rcParams['grid.color'] = 'black'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = True        # Enable grid for axes
plt.rcParams["font.family"] = "Graphik"
plt.rcParams["font.weight"] = "light"
plt.rcParams['figure.figsize'] = [3.5, 2.5]

directories = {
    "boron": ["B1", "B2", "B4"],
    "carbon": ["C1", "C2", "C4"],
    "hydrogen": ["H1", "H2", "H4"],
    "nitrogen": ["N1", "N2", "N4"]
}

element_mapping = {
    "boron": "B",
    "carbon": "C",
    "hydrogen": "H",
    "nitrogen": "N"
}

directory = "06. grain_boundaries"

average_gb_concentrations = {}

for element, contents in directories.items():

    for content in contents:
        data = pd.read_csv(f"{directory}/{element}/CrNi_{content}/outputs/gb_concentration_over_time.csv")
        
        # locate the 8500 "MC Step" and average "Interstitial Concentration" and "Cr Concentration" from 8500 to the final step recorded
        # use final 20% of the step range for averaging
        step_cut = data["MC Step"].max() * 0.8
        data = data[data["MC Step"] >= step_cut]
        interstitial_avg = data["Interstitial Concentration"].mean() * 100
        interstitial_std = data["Interstitial Concentration"].std() * 100
        cr_avg = data["Cr Concentration"].mean() * 100
        cr_std = data["Cr Concentration"].std() * 100

        gb_interstitial_avg = data["GB Interstitial Concentration"].mean() * 100
        gb_interstitial_std = data["GB Interstitial Concentration"].std() * 100
        gb_cr_avg = data["GB Cr Concentration"].mean() * 100
        gb_cr_std = data["GB Cr Concentration"].std() * 100
        
        content_digit = ''.join(filter(str.isdigit, content))
        average_gb_concentrations[f"{element}, {content_digit}"] = {
            "Element": element_mapping[element] if element in element_mapping else element,
            "Content": content_digit,
            "Interstitial Concentration Avg": interstitial_avg,
            "Interstitial Concentration Std": interstitial_std,
            "Cr Concentration Avg": cr_avg,
            "Cr Concentration Std": cr_std,
            "GB Interstitial Concentration Avg": gb_interstitial_avg,
            "GB Interstitial Concentration Std": gb_interstitial_std,
            "GB Cr Concentration Avg": gb_cr_avg,
            "GB Cr Concentration Std": gb_cr_std
        }

df = pd.DataFrame(average_gb_concentrations).T
df.index.name = "Element, Content"
df.to_csv("average_gb_concentration.csv")

# plot interstitial concentration vs. content for each element as a function of content with error bars
for element in ["B", "C", "H", "N"]:
    df_element = df[(df["Element"] == element) & (df["Content"].astype(int) > 0)]
    plt.errorbar(df_element["Content"], df_element["GB Interstitial Concentration Avg"], yerr=df_element["GB Interstitial Concentration Std"], 
                 label=element, linestyle='--',marker='o', linewidth=0.75, capsize=3)

plt.xlabel("Interstitial Content (at%)")
plt.ylabel("Interstitial Concentration (at%)")
plt.ylim(0,15)
#plt.legend()
plt.savefig("plots/compiled_gb_concentration.pdf", dpi=450, bbox_inches='tight')
plt.close()

for element in ["B", "C", "H", "N"]:
    df_element = df[(df["Element"] == element) & (df["Content"].astype(int) > 0)]
    plt.errorbar(df_element["Content"], df_element["Interstitial Concentration Avg"], yerr=df_element["Interstitial Concentration Std"], 
                 label=element, linestyle='--',marker='o', linewidth=0.75, capsize=3)

#plt.xlabel("Interstitial Content (at%)")
#plt.ylabel("Fraction of Interstitials in GB (%)")
plt.ylim(50,100)
#plt.legend()
plt.savefig("plots/compiled_total_gb_concentration.pdf", dpi=450, bbox_inches='tight')