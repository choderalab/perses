import yaml

from perses.utils.url_utils import fetch_url_contents
from perses.utils.url_utils import retrieve_file_url
from perses.tests.utils import enter_temp_directory


class TestData:
    """Test class for data fetching and handling from upstream dataset repository."""
    base_repo_url = 'https://github.com/openforcefield/protein-ligand-benchmark'
    branch_or_tag = '0.2.1'  # release tag
    target_directory = "2020-02-07_tyk2"

    # Test ligands in concatenated file are in the correct order
    def test_ligands_order_concatenated_file(self):
        """Test order is correct for ligands in the concatenated file."""
        from perses.benchmarking.execution import build_ligands_file
        from openff.toolkit.topology import Molecule
        ligands_url = f"{self.base_repo_url}/raw/{self.branch_or_tag}/data/{self.target_directory}/00_data/ligands.yml"
        with enter_temp_directory() as temp_dir:
            with fetch_url_contents(ligands_url) as response:
                ligands_dict = yaml.safe_load(response.read())
            sdf_path = f'{temp_dir}/ligands.sdf'
            build_ligands_file(ligands_dict, path=sdf_path)
            # read molecules from sdf and compare order with dictionary from remote yaml
            molecules_from_file = Molecule.from_file(sdf_path)
            for molecule_yaml_name, molecule_file in zip(ligands_dict.keys(), molecules_from_file):
                molecule_file_name = molecule_file.name
                assert molecule_yaml_name == molecule_file_name, f'Order of molecule not the same. Molecule from ' \
                                                                 f'yaml, {molecule_yaml_name} is not the same as the' \
                                                                 f' one in the sdf file, {molecule_file_name}.'


    def test_protein_file_


    # Test ligands file doesn't get replaced if already exists (based on contents)

    def test_fetch_target_tyk2_pdb(self):
        """Downloads tyk2 coordinate pdb file and checks"""
        from openmm.app.pdbfile import PDBFile
        pdb_url = f"{self.base_repo_url}/raw/{self.branch_or_tag}/data/{self.target_directory}/01_protein/crd/protein.pdb"
        pdb_file = retrieve_file_url(pdb_url)
        omm_pdb = PDBFile(pdb_file)

    def test_fetch_ligands_file_content(self):
        """Fetches information in ligands file for tyk2 and checks the number of ligands is the expected one."""
        ligands_url = f"{self.base_repo_url}/raw/{self.branch_or_tag}/data/{self.target_directory}/00_data/ligands.yml"
        with fetch_url_contents(ligands_url) as response:
            ligands_dict = yaml.safe_load(response.read())
        n_entries = len(ligands_dict)
        assert n_entries == 16, f"Number of fetched ligands is {n_entries}. Expected 16."
