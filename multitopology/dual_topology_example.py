import dualtopology

l1 = "51-61-6"                 # Dopamine: smiles "c1cc(O)c(O)cc1CCN"
l2 = "37491-68-2"              # 3,4-Dihydroxybenzylamine
l3 = "2494-12-4"               # N-Acetyldopamine
ligands = [l1,l2,l3]

dualtop = dualtopology.DualTopology(ligands)
dualtop.createDualTopology()

for atom in dualtop.dual_topology.GetAtoms():
    print(atom)
for bond in dualtop.dual_topology.GetBonds():
    print(bond)
print(dualtop.each_molecule_N)

pdb_filename = "dopamine.pdb"
dualtop.savePDBandFFXML(pdb_filename=pdb_filename)



