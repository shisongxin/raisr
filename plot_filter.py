import argparse

from filterplot import filterplot
import pickle
import numpy as np
from train_denoise import QANGLE, QSTRENGTH, QCOHERENCE, PATCHSIZE

def get_args():
    p = argparse.ArgumentParser(description="Plot denoising filters")
    p.add_argument("-f", "--filter", required=True,
                   help="Path to denoise_sigma*_filter.p model file")
    return p.parse_args()
def main():
    args = get_args()

    # Load model
    print(f"Loading model: {args.filter}")
    with open(args.filter, "rb") as f:
        model = pickle.load(f)
    print(f"  sigma={model['sigma']}, hash_mode={model.get('hash_mode','noisy')}, "
        f"patchsize={model['patchsize']}")
    h_filters = model["h"]  
    h_plot = h_filters[:, :, :, np.newaxis, :]   # (24,3,3,1,121)
    filterplot(h_plot, R=1, Qangle=QANGLE, Qstrength=QSTRENGTH,
                    Qcoherence=QCOHERENCE, patchsize=PATCHSIZE)

if __name__ == "__main__":
    main()