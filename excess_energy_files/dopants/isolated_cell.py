from ase.io import read
from ase.io.lammpsdata import write_lammps_data
import os


def generate_isolated_or_bulk_cells(output_dir="dopant_cells", cell_size=20.0):
    """
    Generate LAMMPS data files for dopants:
    - Reads pre-existing POSCAR files for B and C and converts them to LAMMPS format.
    - Creates diatomic molecules for H and N and saves them in LAMMPS format.

    Parameters:
    - output_dir (str): Directory to save the LAMMPS data files.
    - cell_size (float): Length of the cubic cell for diatomic molecules (Å).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dopant specifications
    dopants = {
        "B": {"form": "bulk"},
        "C": {"form": "bulk"},
        "H": {"form": "diatomic", "bond_length": 0.74},  # Approximate bond length for H2 in Å
        "N": {"form": "diatomic", "bond_length": 1.10},  # Approximate bond length for N2 in Å
    }

    for dopant, properties in dopants.items():
        if properties["form"] == "bulk":
            # Process bulk structures for B and C by reading POSCAR
            poscar_file = f"{output_dir}/{dopant}_POSCAR"
            if not os.path.exists(poscar_file):
                print(f"POSCAR file not found for {dopant}. Skipping.")
                continue

            print(f"Reading bulk structure for {dopant} from {poscar_file}...")
            structure = read(poscar_file, format="vasp")*(2,2,2)

            # Save as LAMMPS data
            lammps_file = os.path.join(output_dir, f"{dopant}_POSCAR.data")
            write_lammps_data(lammps_file, structure)
            print(f"Saved LAMMPS data file for {dopant}: {lammps_file}")

        elif properties["form"] == "diatomic":
            # Create diatomic molecules for H2 and N2
            bond_length = properties["bond_length"]
            atoms = [
                [1, cell_size / 2 - bond_length / 2, cell_size / 2, cell_size / 2],
                [2, cell_size / 2 + bond_length / 2, cell_size / 2, cell_size / 2],
            ]
            atom_count = 2

            # Save LAMMPS data
            lammps_file = os.path.join(output_dir, f"{dopant}_POSCAR.data")
            with open(lammps_file, "w") as f:
                f.write(f"#LAMMPS data file for {dopant}\n\n")
                f.write(f"{atom_count} atoms\n")
                f.write(f"1 atom types\n\n")
                f.write(f"0.0 {cell_size} xlo xhi\n")
                f.write(f"0.0 {cell_size} ylo yhi\n")
                f.write(f"0.0 {cell_size} zlo zhi\n\n")

                # Masses section
                f.write("Masses\n\n")
                f.write("1 1.0\n\n")  # Assign a default mass of 1.0 for simplicity

                # Atoms section
                f.write("Atoms\n\n")
                for i, atom in enumerate(atoms, start=1):
                    f.write(f"{i} 1 {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")
            print(f"Saved LAMMPS data file for {dopant}: {lammps_file}")


# Usage example
generate_isolated_or_bulk_cells()
