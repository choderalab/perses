import yaml
import copy

vacuum_switching_lengths = [0, 100, 500, 1000, 5000, 10000]
solvent_switching_lengths = [500, 1000, 5000, 10000, 20000, 50000]

use_sterics = [True, False]
geometry_divisions = [90, 180, 360, 720]

# Load in template yaml:
with open("rj_hydration.yaml", "r") as templatefile:
    template_yaml = yaml.load(templatefile)

# Load in the template submit script:
with open("submit-template.sh", 'r') as submit_template:
    bsub_template = submit_template.read()

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
                submit_script_filename = "submit-" + specification_file_prefix + ".sh"
                yaml_dict['geometry_divisions'][phase] = geometry_division
                yaml_dict['use_sterics']['vacuum'] = sterics
                yaml_dict['ncmc_switching_times'][phase] = switching_length
                yaml_dict['phase'] = phase
                yaml_dict['output_filename'] = specification_file_prefix + ".nc"

                with open(specification_filename, 'w') as yaml_outfile:
                    yaml.dump(yaml_dict, yaml_outfile)

                with open(submit_script_filename, 'w') as submit_outfile:
                    submit_outfile.write(bsub_template.format(specification_filename))


