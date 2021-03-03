# FEP_molecule_pair

## oe_gen_restricted_confs.py
1. modified from https://github.com/FoldingAtHome/covid-moonshot/blob/master/synthetic-enumeration/02-generate-poses.py

This step is very memory intensive, So I took it apart
1. 'fptype = oegraphsim.OEGetFPType("Tree,ver=2.0.0,size=4096,bonds=%d-%d,atype=AtmNum,btype=Order"
                                    % (minbonds, maxbonds))'
                                    
And I copy some code from http://practicalcheminformatics.blogspot.com/2020/03/building-on-fragments-from-diamondxchem_30.html

## rd_gen_restricted_confs_mpi.py
1. modified from https://github.com/PatWalters/fragment_expansion/blob/master/rdkit_eval/rd_gen_restricted_confs.py
2. add multiple CPU runs
