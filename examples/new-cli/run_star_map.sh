#!/usr/bin/env bash

set -xeuo pipefail

old_ligand_idx=0
for new_ligand_idx in $(seq 1 5)
do 
	perses-cli --yaml my.yaml --override old_ligand_index:"$old_ligand_idx" --override new_ligand_index:"$new_ligand_idx" --override trajectory_directory:lig"$old_ligand_idx"to"$new_ligand_idx"
done
