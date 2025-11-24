import os
import argparse
import pandas as pd

from Bio import SeqIO

import sys
sys.path.append('.')
from detango.utils import get_alphabet


def create_collection_file(protein, sequence_wt):

    alphabet, map_a2i, map_i2a = get_alphabet()

    mutant_list = []
    for i,wt in enumerate(sequence_wt):
        for mt in alphabet:
            if wt != mt:
                mutant_list.append(f'{wt}{i}{mt}')
    df_e = pd.DataFrame({'mutant':mutant_list})
    df_e.to_csv(f'data/{protein}/intermediates/collection_raw.csv')


def esm_score(protein, sequence_wt, cuda):

    os.system(f'python detango/esm_vep.py \
            --model-location esm1v_t33_650M_UR90S_1 esm1v_t33_650M_UR90S_2 esm1v_t33_650M_UR90S_3 esm1v_t33_650M_UR90S_4 esm1v_t33_650M_UR90S_5 \
            --sequence {sequence_wt} \
            --dms-input data/{protein}/intermediates/collection_raw.csv \
            --mutation-col mutant \
            --dms-output data/{protein}/intermediates/collection_esm.csv \
            --offset-idx 0 \
            --scoring-strategy masked-marginals \
            --cuda-device {cuda}')

    df_e = pd.read_csv(f'data/{protein}/intermediates/collection_esm.csv', index_col=0)
    df_e['esm1v'] = df_e[[f'esm1v_t33_650M_UR90S_{i}' for i in range(1,6)]].mean(axis=1)
    # df_e = df_e[['mutant', 'esm1v']]
    df_e.to_csv(f'data/{protein}/intermediates/collection_esm.csv', index=False)


def extract_esm_embedding(protein, cuda, model_seed=1):

    os.system(f'CUDA_VISIBLE_DEVICES={cuda} python detango/esm_embedding.py esm1v_t33_650M_UR90S_{model_seed} \
                data/{protein}/wt.fasta \
                data/{protein} \
                --repr_layers 33 --include mean per_tok')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', '-p', type=str, help='protein name')
    parser.add_argument('--cuda', '-c', type=int, default=0, help='cuda device')
    parser.add_argument('--esm_embedding_model_seed', type=int, default=1, help='the seed for esm embedding model')
    args = parser.parse_args()

    with open(f'data/{args.protein}/wt.fasta') as f:
        for record in SeqIO.parse(f, 'fasta'):
            sequence_wt = str(record.seq)

    create_collection_file(args.protein, sequence_wt)
    esm_score(args.protein, sequence_wt, args.cuda)

if __name__ == '__main__':
    main()
    