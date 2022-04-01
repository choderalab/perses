#!/usr/bin/env bash

for new_ligand_idx in $(seq 1 5)
do 
	echo perses-cli --yaml my.yaml --override old_ligand_index:0 --override new_ligand_index:"$new_ligand_idx" --override trajectory_directory:lig0to"$new_ligand_idx"
done
