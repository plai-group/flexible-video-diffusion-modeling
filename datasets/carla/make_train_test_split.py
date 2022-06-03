import argparse
import glob
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Directory containing all train/test videos.')
args = parser.parse_args()

files = glob.glob(os.path.join(args.dir, 'video_*.pt'))
indexed = {}
for p in files:
    path = Path(p) 
    idx = int(path.stem.split('_')[1])
    indexed[idx] = path
idxs = sorted(indexed.keys())

train_idxs = idxs[:-100]
test_idxs = idxs[-100:]

def make_str(indices, start_i=0):
    s = ',path\n'
    for i, idx in enumerate(indices):
        s += f'{start_i+i},{indexed[idx]}\n'
    return s

train_path = os.path.join(args.dir, 'video_train.csv')
with open(train_path, 'w') as f:
    f.write(make_str(train_idxs))
test_path = os.path.join(args.dir, 'video_test.csv')
with open(test_path, 'w') as f:
    f.write(make_str(test_idxs, start_i=len(train_idxs)))

