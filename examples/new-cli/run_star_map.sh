#!/usr/bin/env bash

center=0
for new_ligand_idx in $(seq 1 5)
do 
	echo perses-cli --yaml my.yaml --override old_ligand_index:"$center" --override new_ligand_index:"$new_ligand_idx" --override trajectory_directory:lig"$center"to"$new_ligand_idx"
done
