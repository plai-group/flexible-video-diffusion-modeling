"""gqn_mazes dataset."""
"""This implementation is based on https://github.com/saeidnp/cwvae/blob/master/datasets/gqn_mazes/gqn_mazes.py and
   https://github.com/iShohei220/torch-gqn/blob/c0156c72f4e63ca6523ab8d9a6f6b3ce9e0e391d/dataset/convert2torch.py"""

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

from pathlib import Path

_DESCRIPTION = """
# GQN Mazes Dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```
@article {Eslami1204,
	title = {Neural scene representation and rendering},
	author = {Eslami, S. M. Ali and Jimenez Rezende, Danilo and Besse, Frederic and Viola, Fabio and Morcos, Ari S. and Garnelo, Marta and Ruderman, Avraham and Rusu, Andrei A. and Danihelka, Ivo and Gregor, Karol and Reichert, David P. and Buesing, Lars and Weber, Theophane and Vinyals, Oriol and Rosenbaum, Dan and Rabinowitz, Neil and King, Helen and Hillier, Chloe and Botvinick, Matt and Wierstra, Daan and Kavukcuoglu, Koray and Hassabis, Demis},
	doi = {10.1126/science.aar6170},
	publisher = {American Association for the Advancement of Science},
	URL = {https://science.sciencemag.org/content/360/6394/1204},
	journal = {Science},
	year = {2018},
}
```
"""

_CITATION = """
@article{saxena2021clockwork,
      title={Clockwork Variational Autoencoders}, 
      author={Vaibhav Saxena and Jimmy Ba and Danijar Hafner},
      year={2021},
      eprint={2102.09532},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DOWNLOAD_URL = "https://archive.org/download/gqn_mazes/gqn_mazes.zip"


class GqnMazes(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for GQN Mazes dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "video": tfds.features.Video(shape=(None, 64, 64, 3)),
                }
            ),
            supervised_keys=None,
            homepage="https://archive.org/details/gqn_mazes",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_DOWNLOAD_URL)

        return {
            "train": self._generate_examples(path / "train"),
            "test": self._generate_examples(path / "test"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob("*.mp4"):
            yield str(f), {
                "video": str(f.resolve()),
            }


def _process_seq(seq):
    seq = tf.expand_dims(seq, 0)
    seq = tf.cast(seq, tf.float32) / 255.0
    return seq

if __name__ == "__main__":
    data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    orig_dataset = 'gqn_mazes'
    torch_dataset_path = data_dir / f'{orig_dataset}-torch'
    torch_dataset_path.mkdir(exist_ok=True)

    for split in ['train', 'test']:
        torch_split_path = torch_dataset_path / split
        torch_split_path.mkdir(exist_ok=True)

        ds = tfds.load("gqn_mazes", data_dir=str(data_dir), shuffle_files=False)[split]
        for cnt, item in enumerate(ds):
            video = item["video"].numpy()
            np.save(torch_split_path / f"{cnt}.npy", video)

        print(f' [-] {cnt} scenes in the {split} dataset')