import os
import json
import pymatgen as mg
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram

import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

# Set API Key for Materials Project
API_KEY = "FBZSo4a0JBvPEPd3RGAYYNcLMwOPqtXq"
try:
    mpr = MPRester(API_KEY)
except Exception as e:
    raise ValueError(f"Failed to initialize MPRester. Check your API key. Error: {e}")

# List of interstitials and metals to analyze
interstitials = ["B", "C", "H", "N", "O"]
metals = ["Al", "Cr", "Fe", "Mo", "Nb", "Ni", "Ti"]

# Load existing database if available
if os.path.exists("materials_database.json"):
    with open("materials_database.json", "r") as f:
        data = json.load(f)
        database = data.get("entries", {})
        bulk_metal_energies = data.get("bulk_metal_energies", {})
else:
    database = {}
    bulk_metal_energies = {}

def get_structures(interstitial, metal):
    """Query Materials Project for known (interstitial, metal) compounds, including ICSD IDs."""
    entries = mpr.get_entries_in_chemsys(
        [interstitial, metal], 
        inc_structure=True
    )

    return entries

def convex_hull_sweep(entries, tolerance=0.1):
    """Perform convex hull analysis to find the lowest energy compounds within a tolerance."""
    pd = PhaseDiagram(entries)
    stable_entries = [entry for entry in entries if pd.get_e_above_hull(entry) <= tolerance or entry.composition.reduced_formula == "Ni3C"]
    
    # !! keep compounds; get rid of single element entries !!
    stable_entries = [entry for entry in stable_entries if len(entry.composition.elements) > 1]        
    stable_entries.sort(key=lambda x: pd.get_e_above_hull(x))
    return stable_entries

def relax_structure(structure):
    """Use ASE to relax the structure and return the final energy."""
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.repeat((2, 2, 2))
    estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3, model_version="v5.0.0")
    calculator = ASECalculator(estimator)
    atoms.calc = calculator
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    return atoms.get_potential_energy(), atoms

def get_bulk_metal_energy(metal):
    """Create and relax a bulk metal supercell to get per atom energy, avoiding duplicates."""
    if metal in bulk_metal_energies:
        return bulk_metal_energies[metal]
    try:
        bulk_structure = bulk(metal, cubic=True).repeat((2, 2, 2))
    except:
        bulk_structure = bulk(metal, orthorhombic=True).repeat((2, 2, 2))
        
    estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3, model_version="v5.0.0")
    calculator = ASECalculator(estimator)
    bulk_structure.calc = calculator
    opt = BFGS(bulk_structure)
    opt.run(fmax=0.01)
    total_energy = bulk_structure.get_potential_energy()
    bulk_metal_energies[metal] = total_energy / len(bulk_structure)
    return bulk_metal_energies[metal]

def calculate_chemical_potential(comp_energy, atoms, bulk_metal_energy):
    """Calculate the chemical potential of interstitial and metal species."""
    n_interstitial = len([atom for atom in atoms if atom.symbol in interstitials])
    n_metal = len([atom for atom in atoms if atom.symbol not in interstitials])
    
    mu_interstitial = (comp_energy - (n_metal * bulk_metal_energy)) / n_interstitial
    mu_metal = (comp_energy - (n_interstitial * mu_interstitial)) / n_metal
    
    return mu_interstitial, mu_metal

def is_duplicate(entry_id):
    """Check if an entry is already in the database."""
    for interstitial in database:
        for metal in database[interstitial]:
            for data in database[interstitial][metal]:
                if data["material_id"] == entry_id:
                    return True
    return False

def main():
    for interstitial in interstitials:
        if interstitial not in database:
            database[interstitial] = {}
        for metal in metals:
            if metal not in database[interstitial]:
                database[interstitial][metal] = []
            print(f"\nProcessing {interstitial}-{metal} compounds...\n")
            entries = get_structures(interstitial, metal)
            
            # H has only one structure with Mo and it is quite a bit above the hull
            if interstitial == "H" and metal == "Mo":
                stable_entries = convex_hull_sweep(entries,tolerance=0.4)
            
            # To avoid a ton of oxides, use a higher tolerance
            elif interstitial == "O":
                stable_entries = convex_hull_sweep(entries,0.02)
            
            else:
                stable_entries = convex_hull_sweep(entries)
            
            # Get bulk metal energy
            bulk_metal_energy = get_bulk_metal_energy(metal) # eV/atom
            
            
            for entry in stable_entries:
                if not is_duplicate(entry.entry_id):
                    print("\n"+entry.composition.reduced_formula+"\n")
                    structure = entry.structure
                    relaxed_energy, relaxed_atoms = relax_structure(structure)
                    mu_i, mu_m = calculate_chemical_potential(relaxed_energy, relaxed_atoms, bulk_metal_energy)
                    
                    space_group = structure.get_space_group_info()[0]  # Extract space group symbol from structure
                    
                    data = {
                        "material_id": entry.entry_id,  # Store Materials Project designation number
                        "formula": entry.composition.reduced_formula, # Store the Stoichiometry
                        "space_group": space_group,  # Store crystal structure (e.g., "Fm3m")
                        "relaxed_energy": relaxed_energy, # Total energy (eV)
                        "chemical_potential_interstitial": mu_i,
                        "chemical_potential_metal": mu_m
                    }
                    database[interstitial][metal].append(data)
                    # Save database including bulk metal energies
                    with open("materials_database.json", "w") as f:
                        json.dump({"entries": database, "bulk_metal_energies": bulk_metal_energies}, f, indent=4)
            

if __name__ == "__main__":
    main()
