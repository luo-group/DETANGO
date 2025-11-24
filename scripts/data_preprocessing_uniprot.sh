#!/bin/bash

protein="P62993"
cuda_device=1
stability_col="foldx"
cpus=60 # number of cpus to use for foldx

# create data directory
mkdir -p data
mkdir -p data/$protein
mkdir -p data/$protein/intermediates

# download sequence and structure data
curl -o data/$protein/wt.fasta https://rest.uniprot.org/uniprotkb/$protein.fasta
curl -o data/$protein/AF-$protein-F1-model_v6.pdb https://alphafold.ebi.ac.uk/files/AF-$protein-F1-model_v6.pdb

# generate collection of single mutants, compute evolutionary plausibility scores, and collect esm embeddings
python detango/initialize_esm_data.py --protein $protein --cuda $cuda_device

# generate stability scores
if [ "$stability_col" == "foldx" ]; then
    mkdir -p data/$protein/intermediates/foldx
    python detango/initialize_foldx_data.py --protein $protein --cpus $cpus
fi