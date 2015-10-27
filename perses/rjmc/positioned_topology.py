"""
Contains base utility class to generate an openmm System
object from the topology proposal
"""
import simtk.openmm as openmm


class PositionedTopology(object):
    """

    Arguments
    ---------
    system
    topology
    positions
    metadata : dict

    Not yet incorporating parameters from system but will
    """

    def __init__(self, system, topology, positions, metadata):
        self.chains = []
        self.bonds = []
        self.residues = []
        self.atoms = []
        self.periodicBoxVectors = topology._periodicBoxVectors
        self.importTopology(topology)
        for k, atom in enumerate(self.atoms):
            atom.assignPosition(positions[k])
            atom.oldIndex = atom.index        

    def importTopology(self, topology):
        appToPosAtoms = dict()
        self._importing = True
        for chain in topology.chains():
            positionedChain = self.addChain(chain.id)
            for residue in chain.residues():
                positionedResidue = self.addResidue(residue.name, positionedChain, residue.id)
                for atom in residue.atoms():
                    positionedAtom = self.addAtom(atom.name, atom,element, positionedResidue, atom.id)
                    appToPosAtoms[atom] = positionedAtom
        for bond in topology.bonds():
            atom1 = bond[0]
            atom2 = bond[1]
            positionedBond = self.addBond(appToPosAtoms[atom1], appToPosAtoms[atom2])
        self._importing = False

    def addChain(self, id):
        index = self.getNumChains()
        chain = PositionedChain(index, self, id)
        self.chains.append(chain)
        return chain

    def addResidue(self, name, positionedChain, id):
        index = self.getNumResidues()
        residue = PositionedResidue(name, index, positionedChain, id)
        self.residues.append(residue)
        positionedChain.residues.append(residue)
        return residue

    def addAtom(self, name, element, positionedResidue, id):
        if not self._importing:
            self.renumberIndices()
        index = self.getNumAtoms()
        atom = PositionedAtom(name, element, index, positionedResidue, id)
        self.atoms.append(atom)
        positionedResidue.atoms.append(atom)
        return atom

    def addBond(self, atom1, atom2):
        bond = PositionedBond(atom1, atom2)
        atom1.inBonds.append(bond)
        atom2.inBonds.append(bond)
        self.bonds.append(bond)
        return bond

    def getNumChains(self):
        return len(self.chains)

    def getNumResidues(self):
        return len(self.residues)

    def getNumAtoms(self):
        return len(self.atoms)

    def getNumBonds(self):
        return len(self.bonds)

    def deleteAtoms(self, toDelete):
        """
        toDelete : list of PositionedAtoms
        """
        for atom in toDelete:
            atom.residue.atoms.remove(atom)
            self.atoms.remove(atom)
            for bond in atom.inBonds:
                self.bonds.remove(bond)
                if atom == bond.atom1:
                    bond.atom2.inBonds.remove(bond)
                elif atom == bond.atom2:
                    bond.atom1.inBonds.remove(bond)
        self.renumberIndices()

    def renumberIndices(self):
        for k, atom in enumerate(self.atoms):
            atom.index=k

    def generateAtomMap(self):
        self.renumberIndices()
        atom_map = {}
        for atom in self.atoms:
            if atom.oldIndex is not None:
                atom_map[atom.oldIndex] = atom.index
        self.atom_map = atom_map

class PositionedChain(object):
    def __init__(self, index, positionedTopology, id):
        self.index = index
        self.topology = positionedTopology
        self.id = id
        self.residues = []

    def getNumResidues(self):
        return len(self.residues)

class PositionedResidue(object):
    def __init__(self, name, index, positionedChain, id):
        self.name = name
        self.index = index
        self.chain = positionedChain
        self.id = id
        self.atoms = []

    def getNumAtoms(self):
        return len(self.atoms)

class PositionedAtom(object):
    # does this also need to have something addable by system for atom type (probably yes)
    def __init__(self, name, element, index, positionedResidue, id):
    self.name = name
    self.element = element
    self.index = index
    self.residue = positionedResidue
    self.id = id
    self.inBonds = []
    self.position = None
    self.oldIndex = None

    def getNumBonds(self):
        return len(self.inBonds)

    def assignPosition(self, position):
        self.position = position

class PositionedBond(object):
    # also needs "type" that can find match from system
    def __init__(self, atom1, atom2):
        self.atom1 = atom1
        self.atom2 = atom2
        self.k = None
        self.length = None

    def assignParameters(self, k, length):
        self.k = k
        self.length = length
