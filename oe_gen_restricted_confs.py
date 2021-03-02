#!/usr/bin/env python

from docopt import docopt

from rdkit import Chem
from rdkit.Chem import rdFMCS

from openeye.oechem import *
from openeye.oedocking import *
from openeye.oeomega import *
from openeye.oequacpac import *
from tqdm import tqdm
from p_tqdm import p_map
import numpy as np
import os

cmd_str = """Usage:
oe_gen_restricted_confs_20210222.py --prot PROTEIN_FILE --lig LIGAND_FILE --ref REFERENCE_FILE --out OUTPUT_FILE [--mcs 1]

Options:
--prot PROTEIN_FILE protein file
--lig LIGAND_FILE ligand file to be positioned in the active site
--ref REFERENCE_FILE reference ligand file for comparison
--out OUTPUT_FILE output file name
--mcs whether use Maximum Common Substructure Search (1)
"""


def open_files(cmd_input):
    prot_file_name = cmd_input.get("--prot")
    assay_data_filename = cmd_input.get("--lig")
    ref_file_name = cmd_input.get("--ref")
    output_filename = cmd_input.get("--out")
    is_mcs = cmd_input.get("--mcs")

    prot_fs = oemolistream(prot_file_name)
    # lig_fs = oemolistream(lig_file_name)
    # assay_data_filename = 'hd-on-0002_assay_results_merged_average_pIC50_smiles_Series-A_20210213-update1.csv'
    # Load assay data if available

    assayed_molecules = list()
    with oechem.oemolistream(assay_data_filename) as ifs:
        for mol in ifs.GetOEGraphMols():
            assayed_molecules.append( oechem.OEGraphMol(mol) )
        print(f'  There are {len(assayed_molecules)} assayed molecules')

    ref_fs = oemolistream(ref_file_name)
    #out_fs = oemolostream(out_file_name)

    return prot_fs, assayed_molecules, ref_fs, output_filename, is_mcs


def read_molecules(prot_fs, ref_fs):
    prot_mol = OEGraphMol()
    OEReadMolecule(prot_fs, prot_mol)

    ref_mol = OEGraphMol()
    OEReadMolecule(ref_fs, ref_mol)

    return prot_mol, ref_mol


def build_receptor(prot_fs, prot_mol, ref_mol):
    prot_file_name = prot_fs.GetFileName()
    receptor = OEGraphMol()
    base, _ = os.path.splitext(prot_file_name)
    receptor_file_name = base + "_receptor.oeb"
    if os.path.exists(receptor_file_name):
        print(f"Reading receptor from {receptor_file_name}")
        receptor_fs = oemolistream(receptor_file_name)
        OEReadMolecule(receptor_fs, receptor)
    else:
        print(f"Building receptor")
        OEMakeReceptor(receptor, prot_mol, ref_mol)
        ofs = oemolostream(receptor_file_name)
        OEWriteMolecule(ofs, receptor)
        print(f"Writing receptor to {receptor_file_name}")
    return receptor

def GetFragments(mol, minbonds, maxbonds):
    from openeye import oegraphsim
    frags = []
    fptype = oegraphsim.OEGetFPType("Tree,ver=2.0.0,size=4068,bonds=%d-%d,atype=AtmNum,btype=Order"
                                    % (minbonds, maxbonds))
    #fptype = oegraphsim.OEGetFPType(oegraphsim.OEFPType_Path)

    for abset in oegraphsim.OEGetFPCoverage(mol, fptype, True):
        fragatompred = oechem.OEIsAtomMember(abset.GetAtoms())

        frag = oechem.OEGraphMol()
        adjustHCount = True
        oechem.OESubsetMol(frag, mol, fragatompred, adjustHCount)
        oechem.OEFindRingAtomsAndBonds(frag)
        frags.append(oechem.OEGraphMol(frag))

    return frags

def GetCommonFragments(mollist, frags,
                       atomexpr=oechem.OEExprOpts_DefaultAtoms,
                       bondexpr=oechem.OEExprOpts_DefaultBonds):

    corefrags = []

    from rich.progress import track
    #for frag in track(frags, description='Finding common fragments'):
    for frag in frags:
        ss = oechem.OESubSearch(frag, atomexpr, bondexpr)
        if not ss.IsValid():
            print('Is not valid')
            continue

        validcore = True
        for mol in mollist:
            oechem.OEPrepareSearch(mol, ss)
            validcore = ss.SingleMatch(mol)
            if not validcore:
                break

        if validcore:
            corefrags.append(frag)

    return corefrags

def GetCoreFragment(refmol, mols, frags,
                    minbonds=3, maxbonds=200,
                    atomexpr=oechem.OEExprOpts_DefaultAtoms,
                    bondexpr=oechem.OEExprOpts_DefaultBonds):

    #print("Number of molecules = %d" % len(mols))

    #frags = GetFragments(refmol, minbonds, maxbonds)
    if len(frags) == 0:
        oechem.OEThrow.Error("No fragment is enumerated with bonds %d-%d!" % (minbonds, maxbonds))

    commonfrags = GetCommonFragments(mols, frags, atomexpr, bondexpr)
    if len(commonfrags) == 0:
        oechem.OEThrow.Error("No common fragment is found!")

    #print("Number of common fragments = %d" % len(commonfrags))

    core = None
    for frag in commonfrags:
        if core is None or GetFragmentScore(core) < GetFragmentScore(frag):
            core = frag

    return core

def GetFragmentScore(mol):

    score = 0.0
    score += 2.0 * oechem.OECount(mol, oechem.OEAtomIsInRing())
    score += 1.0 * oechem.OECount(mol, oechem.OENotAtom(oechem.OEAtomIsInRing()))

    return score

def expand_stereochemistry(mols):
    """Expand stereochemistry when uncertain

    Parameters
    ----------
    mols : openeye.oechem.OEGraphMol
        Molecules to be expanded

    Returns
    -------
    expanded_mols : openeye.oechem.OEMol
        Expanded molecules
    """
    expanded_mols = list()

    from openeye import oechem, oeomega
    omegaOpts = oeomega.OEOmegaOptions()
    #omegaOpts.SetParameterVisibility("-useGPU", False)
    omega = oeomega.OEOmega(omegaOpts)
    maxcenters = 12
    forceFlip = False
    enumNitrogen = True
    warts = True # add suffix for stereoisomers
    for mol in mols:
        for enantiomer in oeomega.OEFlipper(mol, maxcenters, forceFlip, enumNitrogen, warts):
            enantiomer = oechem.OEMol(enantiomer)
            expanded_mols.append(enantiomer)

    return expanded_mols

class BumpCheck:
    def __init__(self, prot_mol, cutoff=2.0):
        self.near_nbr = oechem.OENearestNbrs(prot_mol, cutoff)
        self.cutoff = cutoff

    def count(self, lig_mol):
        bump_count = 0
        for nb in self.near_nbr.GetNbrs(lig_mol):
            if (not nb.GetBgn().IsHydrogen()) and (not nb.GetEnd().IsHydrogen()):
                bump_count += np.exp(-0.5 * (nb.GetDist() / self.cutoff)**2)
        return bump_count

def get_mcs(ref_mol, mol):
    """Code to find the maximum common substructure between two molecules."""
    ref_smi = oechem.OEMolToSmiles(ref_mol)
    mol_smi = oechem.OEMolToSmiles(mol)
    ref_mol = Chem.MolFromSmiles(ref_smi)
    mol = Chem.MolFromSmiles(mol_smi)
    Chem.rdmolops.Kekulize(ref_mol)
    Chem.rdmolops.Kekulize(mol)
    return rdFMCS.FindMCS([ref_mol, mol], completeRingsOnly=True, matchValences=True).smartsString

def generate_restricted_conformers(receptor, refmol, mol, frags, core_smarts=None):
    """
    Generate and select a conformer of the specified molecule using the reference molecule

    Parameters
    ----------
    receptor : openeye.oechem.OEGraphMol
        Receptor (already prepped for docking) for identifying optimal pose
    refmol : openeye.oechem.OEGraphMol
        Reference molecule which shares some part in common with the proposed molecule
    mol : openeye.oechem.OEGraphMol
        Molecule whose conformers are to be enumerated
    core_smarts : str, optional, default=None
        If core_smarts is specified, substructure will be extracted using SMARTS.
    """
    from openeye import oechem, oeomega

    # DEBUG: For benzotriazoles, truncate refmol
    #core_smarts = 'c1ccc(NC(=O)[C,N]n2nnc3ccccc32)cc1' # prospective
    #core_smarts = 'NC(=O)[C,N]n2nnc3ccccc32' # retrospective

    # Get core fragment
    if core_smarts:
        # Truncate refmol to SMARTS if specified
        #print(f'Trunctating using SMARTS {refmol_smarts}')
        ss = oechem.OESubSearch(core_smarts)
        oechem.OEPrepareSearch(refmol, ss)
        for match in ss.Match(refmol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            break
        #print(f'refmol has {refmol.NumAtoms()} atoms')
    else:
        core_fragment = GetCoreFragment(refmol, [mol], frags)
        oechem.OESuppressHydrogens(core_fragment)
        #print(f'  Core fragment has {core_fragment.NumAtoms()} heavy atoms')
        MIN_CORE_ATOMS = 6
        if core_fragment.NumAtoms() < MIN_CORE_ATOMS:
            return None

    # Create an Omega instance
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)

    # Set the fixed reference molecule
    omegaFixOpts = oeomega.OEConfFixOptions()
    omegaFixOpts.SetFixMaxMatch(10) # allow multiple MCSS matches
    omegaFixOpts.SetFixDeleteH(True) # only use heavy atoms
    omegaFixOpts.SetFixMol(core_fragment)
    #omegaFixOpts.SetFixSmarts(smarts)
    omegaFixOpts.SetFixRMS(0.5)

    atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_Hybridization
    bondexpr = oechem.OEExprOpts_BondOrder | oechem.OEExprOpts_Aromaticity
    omegaFixOpts.SetAtomExpr(atomexpr)
    omegaFixOpts.SetBondExpr(bondexpr)
    omegaOpts.SetConfFixOptions(omegaFixOpts)

    molBuilderOpts = oeomega.OEMolBuilderOptions()
    molBuilderOpts.SetStrictAtomTypes(False) # don't give up if MMFF types are not found
    omegaOpts.SetMolBuilderOptions(molBuilderOpts)

    omegaOpts.SetWarts(False) # expand molecule title
    omegaOpts.SetStrictStereo(False) # set strict stereochemistry
    omegaOpts.SetIncludeInput(False) # don't include input
    omegaOpts.SetMaxConfs(1000) # generate lots of conformers
    #omegaOpts.SetEnergyWindow(10.0) # allow high energies
    #omegaOpts.SetParameterVisibility("-useGPU", False)
    omega = oeomega.OEOmega(omegaOpts)

    from openeye import oequacpac
    if not oequacpac.OEGetReasonableProtomer(mol):
        print('No reasonable protomer found')
        return None

    mol = oechem.OEMol(mol) # multi-conformer molecule

    ret_code = omega.Build(mol)
    if (mol.GetDimension() != 3) or (ret_code != oeomega.OEOmegaReturnCode_Success):
        print(f'Omega failure: {mol.GetDimension()} and {oeomega.OEGetOmegaError(ret_code)}')
        return None

    # Extract poses
    class Pose(object):
        def __init__(self, conformer):
            self.conformer = conformer
            self.clash_score = None
            self.docking_score = None
            self.overlap_score = None

    poses = [ Pose(conf) for conf in mol.GetConfs() ]

    # Score clashes
    bump_check = BumpCheck(receptor)
    for pose in poses:
        pose.clash_score = bump_check.count(pose.conformer)

    # Score docking poses
    from openeye import oedocking
    score = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
    score.Initialize(receptor)
    for pose in poses:
        pose.docking_score = score.ScoreLigand(pose.conformer)

    # Compute overlap scores
    from openeye import oeshape
    overlap_prep = oeshape.OEOverlapPrep()
    overlap_prep.Prep(refmol)
    shapeFunc = oeshape.OEExactShapeFunc()
    shapeFunc.SetupRef(refmol)
    oeshape_result = oeshape.OEOverlapResults()
    for pose in poses:
        tmpmol = oechem.OEGraphMol(pose.conformer)
        overlap_prep.Prep(tmpmol)
        shapeFunc.Overlap(tmpmol, oeshape_result)
        pose.overlap_score = oeshape_result.GetRefTversky()

    # Filter poses based on top 10% of overlap
    poses = sorted(poses, key= lambda pose : pose.overlap_score)
    poses = poses[int(0.9*len(poses)):]

    # Select the best docking score
    import numpy as np
    poses = sorted(poses, key=lambda pose : pose.docking_score)
    pose = poses[0]
    mol.SetActive(pose.conformer)
    oechem.OESetSDData(mol, 'clash_score', str(pose.clash_score))
    oechem.OESetSDData(mol, 'docking_score', str(pose.docking_score))
    oechem.OESetSDData(mol, 'overlap_score', str(pose.overlap_score))

    # Convert to single-conformer molecule
    mol = oechem.OEGraphMol(mol)

    return mol

def has_ic50(mol, col_name):
    """Return True if this molecule has fluorescence IC50 data"""
    from openeye import oechem
    try:
        pIC50 = oechem.OEGetSDData(mol, col_name)
        pIC50 = float(pIC50)
        return True
    except Exception as e:
        return False

def generate_restricted_conformers_star(args):
    return generate_restricted_conformers(*args)


def generate_poses(receptor, refmol, target_molecules, output_filename):
    """
    Parameters
    ----------
    receptor : openeye.oechem.OEGraphMol
        Receptor (already prepped for docking) for identifying optimal pose
    refmol : openeye.oechem.OEGraphMol
        Reference molecule which shares some part in common with the proposed molecule
    target_molecules : list of OEMol
        List of molecules to build
    output_filename : str
        Output filename for generated conformers
    """
    # Expand uncertain stereochemistry
    print('Expanding uncertain stereochemistry...')
    target_molecules = expand_stereochemistry(target_molecules)
    print(f'  There are {len(target_molecules)} target molecules')

    # Identify optimal conformer for each molecule
    with oechem.oemolostream(output_filename) as ofs:
        # Write reference molecule copy
        refmol_copy = oechem.OEGraphMol(refmol)
        oechem.OESetSDData(refmol_copy, 'clash_score', '0.0')
        oechem.OEWriteMolecule(ofs, refmol_copy)

        from rich.progress import track
        #for mol in track(target_molecules, f'Generating poses for {len(target_molecules)} target molecules'):
        from multiprocessing import Pool
        from tqdm import tqdm

        pool = Pool(60)
        print('Generate fragments of reference compound...')

        if is_mcs == "1":
            print('Generating MCSS...')
            #core_list = []
            #for mol in tqdm(target_molecules):
            #     core_list.append(get_mcs(refmol, mol))
            core_list = p_map(get_mcs, [refmol]*len(target_molecules), target_molecules, num_cpus=60)
            args = [ (receptor, refmol, mol, core_list[step]) for step, mol in enumerate(target_molecules)]
        else:
            frags = GetFragments(refmol, 3, 200)
            args = [ (receptor, refmol, mol, frags) for mol in target_molecules ]

        for pose in track(pool.imap_unordered(generate_restricted_conformers_star, args), total=len(args), description='Enumerating conformers...'):
            if pose is not None:
                oechem.OEWriteMolecule(ofs, pose)
        pool.close()
        pool.join()

        #for mol in tqdm(target_molecules):
        #    pose = generate_restricted_conformers(receptor, core_fragment, mol)
        #    if pose is not None:
        #        oechem.OEWriteMolecule(ofs, pose)


if __name__ == '__main__':

    cmd_input = docopt(cmd_str)
    prot_fs, assayed_molecules, ref_fs, output_filename, is_mcs = open_files(cmd_input)

    def read_molecules(prot_fs, ref_fs):
        prot_mol = OEGraphMol()
        OEReadMolecule(prot_fs, prot_mol)

        ref_mol = OEGraphMol()
        OEReadMolecule(ref_fs, ref_mol)

        return prot_mol, ref_mol
    prot_mol, ref_mol = read_molecules(prot_fs, ref_fs)
    receptor = build_receptor(prot_fs, prot_mol, ref_mol)

    # # Read receptor
    # print('Reading receptor...')
    # from openeye import oechem
    # receptor = oechem.OEGraphMol()
    # receptor_filename = 'Rec.oeb.gz'
    # from openeye import oedocking
    # oedocking.OEReadReceptorFile(receptor, receptor_filename)
    # print(f'  Receptor has {receptor.NumAtoms()} atoms.')

    # Read reference fragment with coordinates
    # refmol_filename = 'Ref.sdf'
    # refmol = None
    # with oechem.oemolistream(refmol_filename) as ifs:
    #     for mol in ifs.GetOEGraphMols():
    #         refmol = mol
    #         break
    # if refmol is None:
    #     raise Exception(f'Could not read {refmol_filename}')
    # print(f'Reference molecule has {refmol.NumAtoms()} atoms')


    # Filter series to include only those with IC50s
    filter_IC50 = False
    if filter_IC50:
        print(f'Retaining only molecules with IC50s...')
        assayed_molecules = [ mol for mol in assayed_molecules if has_ic50(mol, 'IC50') ]
        print(f'  There are {len(assayed_molecules)} target molecules')

    pka_norm =True
    if pka_norm:
        from openeye import oemolprop
        filt = oemolprop.OEFilter()
        filt.SetpKaNormalize(True)

        print('Fixing pKa of molecules in pH 7.4...')
        assayed_molecules_tmp = assayed_molecules
        assayed_molecules = []
        for mol in assayed_molecules_tmp:
            filt(mol)
            assayed_molecules.append(mol)

    # Generate poses for all molecules
    generate_poses(receptor, ref_mol, assayed_molecules, output_filename)
