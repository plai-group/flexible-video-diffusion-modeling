import argparse
from pathlib import Path
import random
import numpy as np
import torch

from improved_diffusion.carla_regressor import load_classifier_regressor_like_paper, predict_coords, base_data_transform
from improved_diffusion.test_util import Protect, get_model_results_path, get_eval_run_identifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_path", type=str)
    parser.add_argument("--regressor_path", type=str)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for CARLA regressor.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Prepare samples directory
    args.eval_dir = Path(args.eval_dir)
    (args.eval_dir / 'coords').mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / 'coords'}")

    # Load CARLA regressor
    classifier, regressor = load_classifier_regressor_like_paper(args.classifier_path, args.regressor_path, args.device)
    
    # Videos to do
    paths_to_do = list((args.eval_dir / 'samples').glob('*.npy'))
    random.shuffle(paths_to_do)  # shuffle them to make parallelism work better

    for path in paths_to_do:
        coords_path = args.eval_dir / 'coords' / path.name
        if coords_path.exists():
            continue
        print(f"Predicting coords for {path}")
        raw_video = np.load(path)
        raw_video = raw_video.transpose(0, 2, 3, 1)  # put channel dim to end, as expected for np array
        video = torch.stack([base_data_transform(frame) for frame in raw_video])
        coords = predict_coords(video, classifier, regressor, args.batch_size)
        np.save(coords_path, coords)

    predict_coords(video, classifier, regressor, args.batch_size)
