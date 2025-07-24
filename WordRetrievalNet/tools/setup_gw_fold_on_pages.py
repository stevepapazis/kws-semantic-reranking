import argparse
import pathlib
import numpy as np
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "gw_path",
    metavar="gw-path"
)
parser.add_argument(
    "-v", "--visualize",
    help="visualize the ground truth",
    action="store_true"
)
parser.add_argument(
    "-w", "--workers",
    help="how many workers to use in parallel",
    type=int,
)
parser.add_argument(
    "-a", "--amount",
    help="how many augmented examples to produce",
    type=int
)

args = parser.parse_args()
gw_path = pathlib.Path(args.gw_path)

amount  = f"-a {args.amount}" if args.amount else ""
workers = f"-w {args.workers}" if args.workers else ""
visualize = f"-v" if args.visualize else ""
arguments = " ".join([amount, workers, visualize])

pages = np.arange(20)
rng = np.random.default_rng(seed=0)
rng.shuffle(pages)
np.savetxt(gw_path/"WRN_split.txt", pages, fmt="%u")

for f in range(4):
    test_pages = np.zeros_like(pages, dtype=np.bool_)
    test_pages[pages[5*f:5*(f+1)]] = 1

    val_pages = np.zeros_like(pages, dtype=np.bool_)
    val_pages[pages[(5*(f+1))%20:(5*(f+2))%20]] = 1

    base_path = gw_path/f"cv{f}"
    (base_path/"documents").mkdir(parents=True, exist_ok=True)
    (base_path/"images").mkdir(parents=True, exist_ok=True)
    np.save(base_path/"test_set.npy", test_pages)
    np.save(base_path/"validation_set.npy", val_pages)

    # labels = []

    for doc in sorted((gw_path/"ground_truth").iterdir()):
        if doc.suffix != ".gtp": continue
        shutil.copyfile(doc, base_path/f"documents/{doc.name}")
        content = np.loadtxt(doc, dtype=np.str_)
        content = np.insert(content, 0, doc.stem, axis=1)
        # labels.append(content)

    # labels = np.vstack(labels)

    for img in (gw_path/"pages").iterdir():
        if img.suffix != ".tif": continue
        shutil.copyfile(img, base_path/f"images/{img.name}")

    tools = pathlib.Path(__file__).parent
    cmd = f"python3 {tools}/setup_gw_pages.py {base_path} {arguments}"
    print(cmd)
    os.system(cmd)

