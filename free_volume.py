import re
import csv
import numpy as np
from glob import glob
from ase.io import read
import matplotlib.pyplot as plt
from ovito.io import import_file
from ase.neighborlist import NeighborList
from ovito.modifiers import PolyhedralTemplateMatchingModifier, GrainSegmentationModifier
from ovito.modifiers import VoronoiAnalysisModifier

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

def find_grain_boundary(filename):
    # Load the input simulation file
    pipeline = import_file(filename)
    atoms = read(filename)

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

    # Filter indices based on fractional x, y, and z-coordinates for core indices
    filtered_indices = [
        i for i, frac in zip(boundary_indices, boundary_fractions) 
        if 0.30 <= frac[0] <= 0.70 and 0.30 <= frac[1] <= 0.70 and 0.20 <= frac[2] <= 0.80
    ]

    # Filter indices based only on the z-coordinate for gb_indices
    gb_indices = [
        i for i, frac in zip(boundary_indices, boundary_fractions) 
        if 0.20 <= frac[2] <= 0.80
    ]

    # Update positions and atom types for filtered atoms
    filtered_positions = atoms.positions[filtered_indices]
    filtered_atom_types = atoms.get_atomic_numbers()[filtered_indices]

    # Return the indices of the atoms that were identified in filtered_positions
    return filtered_indices, gb_indices

def voronoi_free_volume(filename, gb_indices, core_indices):
    atoms = read(filename)
    # Load the input simulation file
    pipeline = import_file(filename)

    # Apply Voronoi analysis modifier
    voronoi_modifier = VoronoiAnalysisModifier(
        compute_indices=True,
        use_radii=False,
        edge_threshold=0.1
    )
    pipeline.modifiers.append(voronoi_modifier)

    # Compute the pipeline to apply the modifiers
    data = pipeline.compute()

    # Access Voronoi volume data
    voronoi_volumes = data.particles['Atomic Volume']
    voronoi_coordination = data.particles['Coordination']

    # Filter the voronoi volumes to only include metallic grain boundary
    # atoms
    # Calculate the Voronoi volumes for the grain boundary atoms
    gb_volumes = voronoi_volumes[gb_indices]
    core_volumes = voronoi_volumes[core_indices]

    metal_volume = []
    metal_coord = []
    for idx in gb_indices:
        if atoms[idx].symbol not in ['B', 'C', 'O', 'N', 'H']:
            metal_volume.append(voronoi_volumes[idx])
            metal_coord.append(voronoi_coordination[idx])


    return metal_volume, metal_coord

def pristine_voronoi_free_volume(filename):
    # Load the input simulation file
    pipeline = import_file(filename)

    # Apply Voronoi analysis modifier
    voronoi_modifier = VoronoiAnalysisModifier(
        compute_indices=True,
        use_radii=False,
        edge_threshold=0.1
    )
    pipeline.modifiers.append(voronoi_modifier)

    # Compute the pipeline to apply the modifiers
    data = pipeline.compute()

    # Access Voronoi volume data
    voronoi_volumes = data.particles['Atomic Volume']
    voronoi_coordination = data.particles['Coordination']

    return np.mean(voronoi_volumes), np.mean(voronoi_coordination)

clean_atoms = glob("01. boron/CrNi_B0/structures/*")
pristine_volume = []
pristine_coord = []
for atom in clean_atoms:
    v, c = pristine_voronoi_free_volume(atom)
    pristine_volume.append(v)
    pristine_coord.append(c)

# reference volume from the pristine, undoped, Cr30Ni structure
pristine_volume = np.mean(pristine_volume)
pristine_coord = np.mean(pristine_coord)
print(pristine_volume, pristine_coord)

filename_dict = {
    'B': '06. grain_boundaries/boron/CrNi_B{}/structures/*',
    'C': '06. grain_boundaries/carbon/CrNi_C{}/structures/*',
    'H': '06. grain_boundaries/hydrogen/CrNi_H{}/structures/*',
    'N': '06. grain_boundaries/nitrogen/CrNi_N{}/structures/*'
    }

structures = glob(filename_dict["B"].format(0))
undoped_volumes = []
undoped_coords = []
for structure in structures[:1]:
    filtered_indices, gb_indices = find_grain_boundary(structure)
    gb_volumes, metal_coord = voronoi_free_volume(structure, gb_indices, filtered_indices)
    undoped_volumes.append((np.array(gb_volumes)-pristine_volume)/pristine_volume * 100)
    undoped_coords.append(np.array(metal_coord))

fixed_contents = [1, 2, 4]

# Define output CSV file
csv_filename = "gb_free_volume.csv"

# Initialize CSV with header
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Fixed Content", "Element", "Mean Free Volume", "Std Dev Free Volume"])

# Loop over fixed content values
for fixed_content in fixed_contents:
    compiled_data = {"Undoped" : undoped_volumes, "Undoped_coord" : undoped_coords}

    # Loop through interstitial types and collect free volume data
    for element in ["B", "C", "H", "N"]:
        gb_volume = []
        metal_coords = []
        structures = glob(filename_dict[element].format(fixed_content))
        
        for structure in structures:
            filtered_indices, gb_indices = find_grain_boundary(structure)
            gb_volumes, metal_coord = voronoi_free_volume(structure, gb_indices, filtered_indices)
            gb_volume.append((np.array(gb_volumes) - pristine_volume)/pristine_volume * 100)
            metal_coords.append((np.array(metal_coord) - np.mean(undoped_coords))/np.mean(undoped_coords) * 100)

        
        # Store free volume distribution
        compiled_data[element] = np.concatenate(gb_volume) if gb_volume else np.array([])

        # Store metal coordination
        compiled_data[element + "_coord"] = np.concatenate(metal_coords) if metal_coords else np.array([])

    # Open the CSV file once and write all data
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        for interstitial in ["Undoped"] + ["B", "C", "H", "N"]:
            volume_data = compiled_data.get(interstitial, np.array([]))
            coord_data = compiled_data.get(interstitial + "_coord", np.array([]))

            # Check if the dataset is empty
            if len(volume_data) == 0:
                mean_volume, std_volume = np.nan, np.nan
                mean_coord, std_coord = np.nan, np.nan
            else:
                mean_volume, std_volume = np.mean(volume_data), np.std(volume_data)
                mean_coord, std_coord = np.mean(coord_data), np.std(coord_data)

            writer.writerow([fixed_content, interstitial, mean_volume, std_volume])
            
    print(f"✅ Fixed Content {fixed_content} processed and saved.")
