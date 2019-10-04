from perses import dualtopology

ligands = []
ligands.append("Glycine")
ligands.append("Lysine")
ligands.append("Alanine")

dualtop = dualtopology.DualTopology(ligands)
dualtop.createDualTopology()

print("Atom indeces from each substructure:")
print(dualtop.each_molecule_N)

dualtop.savePDBandFFXML(pdb_filename="3_aminos.pdb")



