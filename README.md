<h1 align="center">DETANGO</h1>

## 💼 Environment

First, clone the GitHub repository.
```bash
git clone https://github.com/kerrding/DETANGO.git
cd DETANGO
```
Then, setup the Python environment for DETANGO. We use pytorch 2.2.0 and torchvision 0.17.0, which can be installed with the compatible version for your CUDA or CPU following the instructions on the official website of PyTorch (https://pytorch.org/).
```bash
# Tested on Ubuntu 24.04
conda create -n detango python=3.9.12
conda activate detango

pip install -r requirements.txt

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```


## 🛠️ Data preprocessing

To train a DETANGO model, we first need to compute the ESM-1v-predicted mutational effects and $\Delta\Delta G$ (or use experimentally-derived abundance values) for all single mutations and collect the ESM-1v's embeddings for protein residues. Please install FoldX locally and pass the path to the executables in the bash file. Using `P62993` as an example, you can execute `scripts/data_preprocessing_uniprot.sh`, which contains the code snippet shown below.

```bash
#!/bin/bash

protein="P62993"
cuda_device=1
stability_col="foldx"
cpus=60 # number of cpus to use for foldx
foldx_path=scripts/foldx_20251231 # revise this path to your local foldx executable

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
    python detango/initialize_foldx_data.py --protein $protein --cpus $cpus --foldx-path $foldx_path
fi
```

Note: it takes FoldX for approximately 40 minutes on 60 CPUs to calculate the $\Delta\Delta G$ for all single mutations.

## 💻 Model training and Inference

After collecting the required data for running DETANGO, we proceeded to train DETANGO models. Using `P62993` as an example, you can execute `scripts/model_training.sh`, which contains the code snippet shown below. Be sure to modify the protein identifier, wild-type sequence, and stability column as appropriate for your dataset.
```bash
#!/bin/bash

protein="P62993"
sequence_wt="MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDIEQVPQQPTYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
stability_col="foldx"

for sample_seed in 0 1 2 3 4 5 6 7 8 9
do
    python detango/train_inference.py --protein $protein --sequence_wt $sequence_wt --stability_col $stability_col --sample_seed $sample_seed
done

python detango/merge_results.py --protein $protein --sequence_wt $sequence_wt --stability_col $stability_col
```


## 📬 Contact
Please submit GitHub issues or contact Kerr Ding (kerrding[at]gatech[dot]edu) and Yunan Luo (yunan[at]gatech[dot]edu) for any questions related to the source code.
