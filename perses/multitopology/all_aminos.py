import dualtopology

ligands = []
ligands.append("Glycine")
ligands.append("Lysine")
ligands.append("Alanine")
ligands.append("Valine")
ligands.append("Leucine")
ligands.append("Isoleucine")
ligands.append("Phenylalanine")
ligands.append("Tyrosine")
ligands.append("Tryptophan")
ligands.append("Serine")
ligands.append("Threonine")
ligands.append("Cysteine")
ligands.append("Methionine")
ligands.append("Asparagine")
ligands.append("Glutamine")
ligands.append("Arginine")
ligands.append("Histidine")
ligands.append("Aspartic Acid")  # Cirpy will remove Hs with the name "Aspartate"
ligands.append("Glutamic Acid")

dualtop = dualtopology.DualTopology(ligands)
dualtop.createDualTopology()

print("Atom indeces from each substructure:")
print(dualtop.each_molecule_N)

dualtop.savePDBandFFXML(pdb_filename="all_aminos.pdb")



