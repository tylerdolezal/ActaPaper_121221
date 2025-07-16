from simulation_data_class import SimulationData

content = [0,1] + [x for x in range(2, 12, 2)] + [20]

data = {0 : ["01. boron/CrNi_B", [1,2], [-0.5, 0.5]],
        1 : ["02. carbon/CrNi_C", content, [-1.5, 0.75]],
        2 : ["03. hydrogen/CrNi_H", content, [-0.5, 0.5]],
        3 : ["04. nitrogen/CrNi_N", content, [-1.5, 0.75]],
        4 : ["06. grain_boundaries/boron/CrNi_B", [0,1,2,4], [-1,1]],
        5 : ["06. grain_boundaries/carbon/CrNi_C", [0,1,2,4]],
        6 : ["06. grain_boundaries/hydrogen/CrNi_H", [0,1,2,4]],
        7 : ["06. grain_boundaries/nitrogen/CrNi_N", [0,1,2,4]],
        8 : ["05. boron_hydrogen/CrNi_BH", [1]],
        9 : ["paper 1/Ti_C", [1]],
        10 : ["CrNiZr/CrNi_B", [1]]}

# Initialize SimulationData
pristine = [0,1,2,3]
gbs = [4,5,6,7]
for system in [0]:
        for c in data[system][1]:
                sim_data = SimulationData(file_path=data[system][0]+f"{c}")

                #sim_data.energy_per_atom()
                #sim_data.energy_over_time()

                #sim_data.structure_types()
                #sim_data.generate_epoch()
                #sim_data.generate_rdf(nbins=50, r_max=5.0)
                #sim_data.save_rdf()
                #sim_data.plot_rdf(c)
                #sim_data.process_mc_statistics()
                if system in [4,5,6,7,10]:
                        sim_data.create_gb_structures()

                        # changes path to gb_data for further processing
                        sim_data.set_gb_path()
                        #sim_data.generate_rdf(nbins=25)
                        #sim_data.save_rdf()
                        #sim_data.plot_rdf()
                        #sim_data.gb_location()
                        #sim_data.gb_concentration()
                        #sim_data.gb_bond_lengths()
                        #sim_data.gb_concentration_over_time()

        sim_data.compute_excess_energy() # how I made the bar plots per reference
        #sim_data.plot_isotherm([300, 1073])
        #sim_data.plot_gb_sro(data[4][2])
        sim_data.plot_sro(data[system][2])
        #sim_data.plot_structure_types()
