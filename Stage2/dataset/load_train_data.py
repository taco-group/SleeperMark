import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.2")


_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)


_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class Fill50k(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=_FEATURES,
            supervised_keys=None,
            homepage="",
            license="",
            citation="",
        )

    def _split_generators(self, dl_manager):
            metadata_path = "./dataset/metadata.jsonl"
            images_dir = "./dataset"

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "metadata_path": metadata_path,
                        "images_dir": images_dir,
                    },
                ),
            ]

    def _generate_examples(self, metadata_path, images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["file_name"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            yield row["file_name"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
            }
