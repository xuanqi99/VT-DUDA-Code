import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class VisDA2017(ImageList):
    """`VisDA-2017 <http://ai.bu.edu/visda-2017/assets/attachments/VisDA_2017.pdf>`_ Dataset

    Args:
        root (str): Root directory of dataset. All downloaded files will be placed under this directory.
        task (str): The task (domain) to create dataset. Choices include ``'Synthetic'``: synthetic images and \
            ``'Real'``: real-world images.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            train/
                aeroplance/
                    *.png
                    ...
            validation/
            image_list/
                train.txt
                validation.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/b25b2b990e8f42e691f0/?dl=1"),
        ("train", "train.tar", "http://csr.bu.edu/ftp/visda17/clf/train.tar"),
        ("validation", "validation.tar", "http://csr.bu.edu/ftp/visda17/clf/validation.tar")
    ]
    image_list = {
        "Synthetic": "image_list/train.txt",
        "Real": "image_list/validation.txt",
    }
    CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

    @classmethod
    def _domain_key(cls, task: str) -> str:
        return "train" if task == "Synthetic" else "validation"

    @classmethod
    def _candidate_rel_paths(cls, task: str):
        domain_key = cls._domain_key(task)
                                                                  
        return [
            os.path.join(domain_key, "image_list.txt"),                                                         
            os.path.join("image_list", f"{domain_key}.txt"),
            f"{domain_key}.txt",
            f"{domain_key}_list.txt",
        ]

    @classmethod
    def _resolve_data_list_file(cls, root: str, task: str) -> str:
        seen = set()
        for rel in cls._candidate_rel_paths(task):
            rel_norm = rel.replace("\\", "/")
            if rel_norm in seen:
                continue
            seen.add(rel_norm)
            candidate = os.path.join(root, rel)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find an image list for task '{task}' under '{root}'. "
            f"Checked: {cls._candidate_rel_paths(task)}"
        )

    @staticmethod
    def _load_image_list_entries(list_file: str):
        entries = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                rel_path = " ".join(parts[:-1])
                label = int(parts[-1])
                entries.append((rel_path, label))
        return entries

    @staticmethod
    def _normalize_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return os.path.abspath(os.path.expanduser(path))

    @classmethod
    def _candidate_roots(cls, root_hint: Optional[str] = None):
        candidates, seen = [], set()

        def _add(path):
            normalized = cls._normalize_path(path)
            if normalized and normalized not in seen:
                seen.add(normalized)
                candidates.append(normalized)

        _add(root_hint)
        _add(os.environ.get("VISDA2017_ROOT"))

        search_bases = []
        base = os.path.abspath(os.path.dirname(__file__))
        for _ in range(8):
            search_bases.append(base)
            parent = os.path.dirname(base)
            if parent == base:
                break
            base = parent
        search_bases.append(os.getcwd())
        project_root = os.environ.get("PROJECT_ROOT")
        if project_root:
            search_bases.append(project_root)
        search_bases.append(os.path.expanduser("~"))

        for base in search_bases:
            if base.endswith("visda2017"):
                _add(base)
            _add(os.path.join(base, "dataset", "visda2017"))
            _add(os.path.join(base, "datasets", "visda2017"))

        return candidates

    @classmethod
    def _resolve_root(cls, root_hint: Optional[str], download: bool) -> str:
        normalized = cls._normalize_path(root_hint)
        if download:
            target = normalized or cls._normalize_path(os.environ.get("VISDA2017_ROOT"))
            if not target:
                default_candidates = cls._candidate_roots(None)
                target = default_candidates[0] if default_candidates else os.path.join(os.getcwd(), "dataset", "visda2017")
            os.makedirs(target, exist_ok=True)
            return target

        if normalized and os.path.isdir(normalized):
            return normalized

        for candidate in cls._candidate_roots(normalized):
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            f"VisDA2017 root not found for '{root_hint}'. Checked: {cls._candidate_roots(normalized)}"
        )

    @classmethod
    def _pick_image_root(cls, dataset_root: str, task: str, data_list_file: str) -> str:
        """
        Pick the correct image root for ImageList.

        Some VisDA lists (e.g. `validation/image_list.txt`) store paths relative to the
        split folder (`validation/`), not relative to `dataset_root/`.
        """
        domain_key = cls._domain_key(task)
        split_root = os.path.join(dataset_root, domain_key)

                                                                
        entries = cls._load_image_list_entries(data_list_file)
        sample = entries[:50] if len(entries) > 50 else entries

        def _score(root_candidate: str) -> int:
            score = 0
            for rel_path, _ in sample:
                p = rel_path
                                                 
                if os.path.isabs(p):
                    if os.path.isfile(p):
                        score += 1
                else:
                    if os.path.isfile(os.path.join(root_candidate, p)):
                        score += 1
            return score

        score_dataset = _score(dataset_root)
        score_split = _score(split_root)

                                                    
        return split_root if score_split > score_dataset else dataset_root

    @classmethod
    def _normalize_task(cls, task: str) -> str:
        if not isinstance(task, str):
            return task
        t = task.strip()
        low = t.lower()
        if low == "real":
            return "Real"
        if low == "synthetic":
            return "Synthetic"
        return t

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        task = self._normalize_task(task)
        assert task in self.image_list
        self.task = task                                         
        dataset_root = self._resolve_root(root, download)

        def _image_list_present():
            try:
                self._resolve_data_list_file(dataset_root, task)
                return True
            except FileNotFoundError:
                return False

        def _all_present():
                                                                 
            for file_name, _, _ in self.download_list:
                if file_name == "image_list":
                    if not _image_list_present():
                        return False
                    continue
                try:
                    check_exits(dataset_root, file_name)
                except FileNotFoundError:
                    return False
            return True

        if download and not _all_present():
            list(map(lambda args: download_data(dataset_root, *args), self.download_list))
        else:
            for entry in self.download_list:
                if entry[0] == "image_list":
                    if not _image_list_present():
                        raise FileNotFoundError(
                            f"Image list for task '{task}' not found under '{dataset_root}'."
                        )
                    continue
                check_exits(dataset_root, entry[0])

        data_list_file = self._resolve_data_list_file(dataset_root, task)

                                                                                 
        image_root = self._pick_image_root(dataset_root, task, data_list_file)

        super(VisDA2017, self).__init__(
            image_root, VisDA2017.CLASSES, data_list_file=data_list_file, **kwargs
        )
        self.data_list = self._load_image_list_entries(data_list_file)

                                                                                 
                                             
        self.dataset_root = dataset_root
        self.data_list_file = data_list_file
        self.image_root = image_root                                              

    def __repr__(self) -> str:
        data_list = getattr(self, "data_list", None)
        n = (len(data_list) if isinstance(data_list, list) else None) or len(self)
        return (f"VisDA2017(task={getattr(self, 'task', None)!r}, "
                f"dataset_root={getattr(self, 'dataset_root', None)!r}, "
                f"image_root={getattr(self, 'image_root', None)!r}, "
                f"data_list_file={getattr(self, 'data_list_file', None)!r}, "
                f"n={n})")

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

    @classmethod
    def num_classes(cls) -> int:
        return len(cls.CLASSES)

    @classmethod
    def class_names(cls):
        return list(cls.CLASSES)

    @classmethod
    def classes(cls):
        """Alias for callers that expect a 'classes()' accessor."""
        return list(cls.CLASSES)

    @classmethod
    def class_count(cls) -> int:
        """Alias for callers that prefer a 'class_count()' name."""
        return len(cls.CLASSES)

    @classmethod
    def sanity_check_classes(cls) -> None:
        """Optional: verify class list is consistent (no-op unless called)."""
        if not isinstance(cls.CLASSES, (list, tuple)) or len(cls.CLASSES) != 12:
            raise RuntimeError(f"VisDA2017.CLASSES must have 12 entries, got {len(getattr(cls, 'CLASSES', []))}")
