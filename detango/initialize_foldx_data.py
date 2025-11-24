import argparse
import os
from multiprocessing import Pool
import pandas as pd

from tqdm import tqdm

import time

from Bio import SeqIO


def repair_pdb(foldx_path, pdb_name, input_path, output_path):
    os.system(f'{foldx_path} --command=RepairPDB \
                    --pdb={pdb_name}.pdb \
                    --pdb-dir={input_path} \
                    --output-dir={output_path}')


def compute_foldx_stability_single(data):
    foldx_path, pdb_name, output_path, i = data

    cmd = f'{foldx_path} \
            --command=BuildModel \
            --pdb={pdb_name}_Repair.pdb \
            --pdb-dir={output_path} \
            --mutant-file={output_path}/{i}/individual_list.txt \
            --numberOfRuns=5 \
            --output-dir={output_path}/{i}/ \
            --out-pdb=False'
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', '-p', type=str, help='protein name')
    parser.add_argument('--pdb_name', type=str, help='name of the pdb file, default is AF-{protein}-F1-model_v6')
    parser.add_argument('--cpus', type=int, default=60, help='number of cpus to use for foldx')
    parser.add_argument('--foldx-path', type=str, default='scripts/foldx_20251231', help='path to foldx executable')
    args = parser.parse_args()

    # st = time.time()

    with open(f'data/{args.protein}/wt.fasta') as f:
        sequence_wt = str(next(SeqIO.parse(f, 'fasta')).seq)
    
    # Repair the PDB file using FoldX
    pdb_name = args.pdb_name if args.pdb_name else f'AF-{args.protein}-F1-model_v6'
    input_path = f'data/{args.protein}'
    output_path = f'data/{args.protein}/intermediates/foldx'
    repair_pdb(args.foldx_path, pdb_name, input_path, output_path)

    # Generate single mutants
    df_e = pd.read_csv(f'data/{args.protein}/intermediates/collection_raw.csv')
    workload = []
    for i,mut in tqdm(enumerate(df_e.mutant.values)):
        os.system(f'mkdir -p {output_path}/{i}')
        with open(f'{output_path}/{i}/individual_list.txt', 'w') as f:
            wt, pos, mt = mut[0], int(mut[1:-1])+1, mut[-1]
            f.write(wt+'A'+str(pos)+mt+';')
        workload.append((args.foldx_path, pdb_name, output_path, i))

    # Compute stability changes using FoldX
    with Pool(processes=args.cpus) as p:
        p.map(compute_foldx_stability_single, workload, chunksize=1)

    # ed = time.time()
    # print(f'FoldX computation time: {ed-st:.1f} seconds')

    # collect foldx results
    dfs = []
    for i in range(len(df_e)):
        with open(f'{output_path}/{i}/individual_list.txt', 'r') as f:
            input_list = f.readlines()
        input_list = [x.strip() for x in input_list]

        df = pd.read_csv(f'{output_path}/{i}/Average_{pdb_name}_Repair.fxout',
                            header=7, delimiter='\t')
        df = df.drop_duplicates('Pdb', keep='last')
        assert len(df) == len(input_list)
        df['input'] = input_list
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates('input', keep='first')
    df.to_csv(f'/work/kerr/disentangle/DETANGO/data/{args.protein}/intermediates/foldx_results.csv', index=False)
    assert len(df) == len(sequence_wt) * 19

    # merge results back to collection file
    df_e = pd.read_csv(f'data/{args.protein}/intermediates/collection_esm.csv')
    df['mutant'] = df['input'].apply(lambda x: x[0]+str(int(x[2:-2])-1)+x[-2])
    df['foldx'] = df['total energy']
    df_e = df_e.merge(df[['mutant', 'foldx']], on='mutant', how='left')
    df_e.to_csv(f'data/{args.protein}/intermediates/collection.csv', index=False)



if __name__ == '__main__':
    main()