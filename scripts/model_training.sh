#!/bin/bash

protein="P62993"
sequence_wt="MEAIAKYDFKATADDELSFKRGDILKVLNEECDQNWYKAELNGKDGFIPKNYIEMKPHPWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDIEQVPQQPTYVQALFDFDPQEDGELGFRRGDFIHVMDNSDPNWWKGACHGQTGMFPRNYVTPVNRNV"
stability_col="foldx"

for sample_seed in 0 1 2 3 4 5 6 7 8 9
do
    python detango/train_inference.py --protein $protein --sequence_wt $sequence_wt --stability_col $stability_col --sample_seed $sample_seed
done

python detango/merge_results.py --protein $protein --sequence_wt $sequence_wt --stability_col $stability_col