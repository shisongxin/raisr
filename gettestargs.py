import argparse
import os


def gettestargs(argv=None):
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filter", default=os.path.join(root, "filter_aligned.p"))
    parser.add_argument("-p", "--plot", action="store_true", help="Plot upscaling results")
    parser.add_argument("--input-dir", default=os.path.join(root, "test"))
    parser.add_argument("--output-dir", default=os.path.join(root, "results"))
    return parser.parse_args(argv)
