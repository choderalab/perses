import dualtopology

l1 = "Glycine"
l2 = "Lysine"
ligands = [l1,l2]

dualtop = dualtopology.DualTopology(ligands)
dualtop.createDualTopology()

print("Atoms in dual topology:")
for atom in dualtop.dual_topology.GetAtoms():
    print(atom)
print("Bonds in dual topology:")
for bond in dualtop.dual_topology.GetBonds():
    print(bond)
print("Atom indeces from each substructure:")
print(dualtop.each_molecule_N)

dualtop.savePDBandFFXML()



