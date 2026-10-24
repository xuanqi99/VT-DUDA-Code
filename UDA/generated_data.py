"""Optional image-list inputs shared by the existing UDA training scripts.

This module keeps dataset preparation independent of PyTorch so the experiment
grid can be inspected on a machine without the training environment installed.
Labels follow the CLASSES order in the supplied dataset implementations.
"""

import ast
from pathlib import Path


DATASET_FILES = {
    "OfficeHome": "officehome.py",
    "Office31": "office31.py",
    "VisDA2017": "visda2017.py",
}


def dataset_classes(dataset_name):
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"List inputs support: {', '.join(DATASET_FILES)}")
    source = Path(__file__).parent / "ELS" / "common" / "vision" / "datasets" / DATASET_FILES[dataset_name]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CLASSES" for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
                return names
    raise ValueError(f"No class definition found in {source}")


def read_image_list(list_file, image_root, num_classes, check_files=True):
    """Read paths (including spaces) followed by an integer class label."""
    root = Path(image_root).expanduser().resolve()
    samples = []
    for line_number, raw in enumerate(Path(list_file).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            image, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(f"{list_file}:{line_number}: expected 'image_path integer_label'") from exc
        if not 0 <= label < num_classes:
            raise ValueError(f"{list_file}:{line_number}: label {label} outside [0, {num_classes - 1}]")
        path = Path(image).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if check_files and not path.is_file():
            raise FileNotFoundError(f"{list_file}:{line_number}: missing image {path}")
        samples.append((str(path), label))
    if not samples:
        raise ValueError(f"Empty image list: {list_file}")
    return samples


class ImageListDataset:
    """Picklable Dataset protocol with the same sample structure as existing code."""

    def __init__(self, samples, classes, transform, domain_id=None):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform
        self.domain_id = domain_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        sample = (image, label)
        return sample if self.domain_id is None else sample + (self.domain_id,)


def add_list_arguments(parser):
    parser.add_argument("--source-list", nargs="+", default=None,
                        help="Optional labeled source image lists; replaces domain-based loading.")
    parser.add_argument("--target-list", default=None,
                        help="Labeled real target list for adaptation and evaluation (labels ignored during adaptation).")
    parser.add_argument("--target-image-root", default=None,
                        help="Root for relative target list paths; defaults to positional dataset root.")


def list_inputs_requested(args):
    requested = bool(args.source_list or args.target_list)
    if requested and not (args.source_list and args.target_list):
        raise ValueError("--source-list and --target-list must be supplied together")
    return requested


def get_list_datasets(args, train_transform, val_transform, with_domain_id=False):
    from torch.utils.data import ConcatDataset

    if getattr(args, "dataset_ratio", -1) != -1:
        raise ValueError("Image-list mode uses the complete source lists; --dataset-ratio must be -1")
    names = dataset_classes(args.data)
    source = []
    for index, list_file in enumerate(args.source_list):
        samples = read_image_list(list_file, args.root, len(names))
        source.append(ImageListDataset(samples, names, train_transform, index if with_domain_id else None))
    target_samples = read_image_list(args.target_list, args.target_image_root or args.root, len(names))
    domain_id = len(source) if with_domain_id else None
    train_target = ImageListDataset(target_samples, names, train_transform, domain_id)
    evaluation = ImageListDataset(target_samples, names, val_transform, domain_id)
    return ConcatDataset(source), train_target, evaluation, evaluation, len(names), names
