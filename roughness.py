import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

relrms = 0.0

data = {1 : ["06. grain_boundaries/boron/CrNi_B", [0,1,2,4], [-1,1]],
        2 : ["06. grain_boundaries/carbon/CrNi_C", [1,2,4]],
        3 : ["06. grain_boundaries/hydrogen/CrNi_H", [1,2,4]],
        4 : ["06. grain_boundaries/nitrogen/CrNi_N", [1,2,4]]}

with open("roughness.csv", "w") as f:
    f.write("file, rms_roughness (Å)\n")


# Build file paths and loop over them
for key, (base_path, dopant_levels, *_) in data.items():
    for level in dopant_levels:
        file = f"{base_path}{level}/structures/POSCARgb-1"
    
        # Step 1: Load atomic positions from POSCAR-B1
        atoms = read(file)  # Load the structure using ASE
        positions = atoms.get_positions()  # Extract atomic positions (x, y, z)
        symbols = np.array(atoms.get_chemical_symbols())  # Atomic symbols
        # Filter positions for atoms whose symbol is NOT "B" or "C"
        positions = positions[~np.isin(symbols, ["B", "C"])]
        # Step 2: Define the binning parameters
        nbins = 5  # Number of bins along each axis
        x_edges = np.linspace(np.min(positions[:, 0]), np.max(positions[:, 0]), nbins + 1)
        y_edges = np.linspace(np.min(positions[:, 1]), np.max(positions[:, 1]), nbins + 1)

        # Step 3: Digitize the \(x\)-\(y\) space
        x_indices = np.digitize(positions[:, 0], x_edges) - 1
        y_indices = np.digitize(positions[:, 1], y_edges) - 1

        # Step 4: Compute the average \(z\)-coordinate for each bin
        z_bins = np.zeros((nbins, nbins))  # Store average \(z\) values
        counts = np.zeros((nbins, nbins))  # Store atom counts for each bin

        for x_idx, y_idx, z in zip(x_indices, y_indices, positions[:, 2]):
            if 0 <= x_idx < nbins and 0 <= y_idx < nbins:
                z_bins[x_idx, y_idx] += z
                counts[x_idx, y_idx] += 1

        # Avoid division by zero (bins with no atoms)
        z_bins[counts > 0] /= counts[counts > 0]

        # Step 5: Generate a grid for visualization
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])  # Bin centers along x
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])  # Bin centers along y
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)

        # Step 6: Plot the surface
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(x_grid, y_grid, z_bins.T, edgecolor="k", alpha=0.65)

        # Step 7: Fix the view to look at the x-z plane
        ax.view_init(elev=0, azim=90)  # Elevation = 0, Azimuth = 90 for X-Z view

        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        ax.set_zlim(0.4, 0.6)
        ax.set_zlim(0, atoms.get_cell()[2, 2])
        # Turn off the axes and grid
        #ax.set_axis_off()  # Completely turn off the axes and grid
        #plt.savefig(f"{file}.png", dpi=400, bbox_inches='tight', transparent=True)

        z_mean = np.mean(positions[:, 2]) # Mean surface height
        rms_roughness = np.sqrt(np.nanmean((positions[:, 2] - z_mean)**2))
        rms_roughness -= relrms

        if base_path+str(level) == f"{base_path}"+"0":
            relrms = rms_roughness
        
        # save z_mean and rms_roughness to csv file
        with open("roughness.csv", "a") as f:
            f.write(f"{file.split('_')[-1:][0].split('/')[0]},{rms_roughness}\n")

