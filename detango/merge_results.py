import pandas as pd
import argparse
import os
import pickle

import sys
sys.path.append('.')
from detango.utils import transform_df_to_matrix

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', '-p', type=str, help='protein name')
    parser.add_argument('--sequence_wt', type=str, help='wild-type sequence')
    parser.add_argument('--stability_col', type=str, default='foldx', help='the column name for stability in the csv file')
    args = parser.parse_args()

    dfs = []
    for seed in range(10):
        df = pd.read_csv(f'results/{args.protein}/intermediates/detango_{args.stability_col}_{seed}.csv')
        dfs.append(df)

    df_var = dfs[0][['mutant']]
    df_var['detango'] = 0

    for df in dfs:
        df_var['detango'] += df['q']
    df_var['detango'] /= len(dfs)

    df_var.to_csv(f'results/{args.protein}/detango_variants_{args.stability_col}_ensemble.csv', index=False)

    residue_scores = transform_df_to_matrix(df_var, args.sequence_wt, 'detango', shift=0, fillna=0, col_mut='mutant', flip=True)
    df_res = pd.DataFrame({
        'residue': [f'{aa}{i}' for i,aa in enumerate(args.sequence_wt)],
        'detango': residue_scores.mean(axis=1)
    })

    df_res.to_csv(f'results/{args.protein}/detango_residues_{args.stability_col}_ensemble.csv', index=False)


if __name__ == "__main__":
    main()