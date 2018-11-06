import yaml
import copy

vacuum_switching_lengths = [0, 100, 500, 1000, 5000, 10000]
solvent_switching_lengths = [500, 1000, 5000, 10000, 20000, 50000]

use_sterics = [True, False]
geometry_divisions = [90, 180, 360, 720]

# Load in template yaml:
with open("rj_hydration.yaml", "r") as templatefile:
    template_yaml = yaml.load(templatefile)


# Set up vacuum simulations:
for phase in ['vacuum', 'explicit']:
    if phase == "vacuum":
        switching_lengths = vacuum_switching_lengths
    else:
        switching_lengths = solvent_switching_lengths

    for switching_length in switching_lengths:
        for sterics in use_sterics:
            for geometry_division in geometry_divisions:
                yaml_dict = copy.deepcopy(template_yaml)
                specification_file_prefix = "{}_{}ncmc_{}sterics_{}geometry".format(phase, switching_length, sterics, geometry_division)
                specification_filename = specification_file_prefix + ".yaml"
                yaml_dict['geometry_divisions'][phase] = geometry_division
                yaml_dict['use_sterics']['vacuum'] = sterics
                yaml_dict['ncmc_switching_times'][phase] = switching_length
                yaml_dict['phase'] = phase
                yaml_dict['output_filename'] = specification_file_prefix + ".nc"

                with open(specification_filename, 'w') as yam_outfile:
                    yaml.dump(yaml_dict, yam_outfile)
