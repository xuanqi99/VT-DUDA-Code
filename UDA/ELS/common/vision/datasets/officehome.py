import os
import random
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
        "output_Ar2Cl": "image_list/output_Ar2Cl.txt",
        "output_R2C_generate": "image_list/output_R2C_generate.txt",
        "output_R2P_generate": "image_list/output_R2P_generate.txt",
        "output_A2C_generate": "image_list/output_A2C_generate.txt",
        "output_A2P_generate": "image_list/output_A2P_generate.txt",
        "output_A2R_generate": "image_list/output_A2R_generate.txt",
        "RealWorldCustom": "image_list/Real World.txt",
        "output_C2A_generate": "image_list/output_C2A_generate.txt",
        "output_C2P_generate": "image_list/output_C2P_generate.txt",
        "output_C2R_generate": "image_list/output_C2R_generate.txt",
        "output_P2A_generate": "image_list/output_P2A_generate.txt",
        "output_P2C_generate": "image_list/output_P2C_generate.txt",
        "output_P2R_generate": "image_list/output_P2R_generate.txt",
        "output_R2A_generate": "image_list/output_R2A_generate.txt",
                                             
        "output_A2Ctoken_generate": "image_list/output_A2Ctoken_generate.txt",
        "output_A2Ptoken_generate": "image_list/output_A2Ptoken_generate.txt",
        "output_A2Rtoken_generate": "image_list/output_A2Rtoken_generate.txt",
        "output_C2Atoken_generate": "image_list/output_C2Atoken_generate.txt",
        "output_C2Ptoken_generate": "image_list/output_C2Ptoken_generate.txt",
        "output_C2Rtoken_generate": "image_list/output_C2Rtoken_generate.txt",
        "output_P2Atoken_generate": "image_list/output_P2Atoken_generate.txt",
        "output_P2Ctoken_generate": "image_list/output_P2Ctoken_generate.txt",
        "output_P2Rtoken_generate": "image_list/output_P2Rtoken_generate.txt",
        "output_R2Atoken_generate": "image_list/output_R2Atoken_generate.txt",
        "output_R2Ctoken_generate": "image_list/output_R2Ctoken_generate.txt",
        "output_R2Ptoken_generate": "image_list/output_R2Ptoken_generate.txt",
                                                         
        "output_OfficeHome-CRconcept_token20_generate": "image_list/output_OfficeHome-CRconcept_token20_generate.txt",
                                                                 
        "output_OfficeHome-CRconcept_token5_generate": "image_list/output_OfficeHome-CRconcept_token5_generate.txt",
        "output_OfficeHome-CRconcept_token50_generate": "image_list/output_OfficeHome-CRconcept_token50_generate.txt",
    }
                                                 
    TASK_ALIASES = {
        "ar": "Ar", "art": "Ar", "a": "Ar",
        "cl": "Cl", "clipart": "Cl", "c": "Cl",
        "pr": "Pr", "product": "Pr", "p": "Pr",
        "rw": "Rw", "real_world": "Rw", "real-world": "Rw", "realworld": "Rw",
        "real world": "RealWorldCustom",
        "real": "Rw", "r": "Rw",
        "output_ar2cl": "output_Ar2Cl",
        "output_r2c_generate": "output_R2C_generate",
        "output_r2p_generate": "output_R2P_generate",
        "output_a2c_generate": "output_A2C_generate",
        "a2c_generate": "output_A2C_generate",
        "output_a2c": "output_A2C_generate",
        "output_a2p_generate": "output_A2P_generate",
        "a2p_generate": "output_A2P_generate",
        "output_a2p": "output_A2P_generate",
        "output_a2r_generate": "output_A2R_generate",
        "a2r_generate": "output_A2R_generate",
        "output_a2r": "output_A2R_generate",
                                                      
        "output_c2a_generate": "output_C2A_generate",
        "c2a_generate": "output_C2A_generate",
        "output_c2a": "output_C2A_generate",
        "output_c2p_generate": "output_C2P_generate",
        "c2p_generate": "output_C2P_generate",
        "output_c2p": "output_C2P_generate",
        "output_c2r_generate": "output_C2R_generate",
        "c2r_generate": "output_C2R_generate",
        "output_c2r": "output_C2R_generate",
        "output_p2a_generate": "output_P2A_generate",
        "p2a_generate": "output_P2A_generate",
        "output_p2a": "output_P2A_generate",
        "output_p2c_generate": "output_P2C_generate",
        "p2c_generate": "output_P2C_generate",
        "output_p2c": "output_P2C_generate",
        "output_p2r_generate": "output_P2R_generate",
        "p2r_generate": "output_P2R_generate",
        "output_p2r": "output_P2R_generate",
        "output_r2a_generate": "output_R2A_generate",
        "r2a_generate": "output_R2A_generate",
        "output_r2a": "output_R2A_generate",
                                                                  
        "output_a2ctoken_generate": "output_A2Ctoken_generate",
        "a2ctoken_generate": "output_A2Ctoken_generate",
        "output_a2ctoken": "output_A2Ctoken_generate",
        "a2ctoken": "output_A2Ctoken_generate",

        "output_a2ptoken_generate": "output_A2Ptoken_generate",
        "a2ptoken_generate": "output_A2Ptoken_generate",
        "output_a2ptoken": "output_A2Ptoken_generate",
        "a2ptoken": "output_A2Ptoken_generate",

        "output_a2rtoken_generate": "output_A2Rtoken_generate",
        "a2rtoken_generate": "output_A2Rtoken_generate",
        "output_a2rtoken": "output_A2Rtoken_generate",
        "a2rtoken": "output_A2Rtoken_generate",

        "output_c2atoken_generate": "output_C2Atoken_generate",
        "c2atoken_generate": "output_C2Atoken_generate",
        "output_c2atoken": "output_C2Atoken_generate",
        "c2atoken": "output_C2Atoken_generate",

        "output_c2ptoken_generate": "output_C2Ptoken_generate",
        "c2ptoken_generate": "output_C2Ptoken_generate",
        "output_c2ptoken": "output_C2Ptoken_generate",
        "c2ptoken": "output_C2Ptoken_generate",

        "output_c2rtoken_generate": "output_C2Rtoken_generate",
        "c2rtoken_generate": "output_C2Rtoken_generate",
        "output_c2rtoken": "output_C2Rtoken_generate",
        "c2rtoken": "output_C2Rtoken_generate",

        "output_p2atoken_generate": "output_P2Atoken_generate",
        "p2atoken_generate": "output_P2Atoken_generate",
        "output_p2atoken": "output_P2Atoken_generate",
        "p2atoken": "output_P2Atoken_generate",

        "output_p2ctoken_generate": "output_P2Ctoken_generate",
        "p2ctoken_generate": "output_P2Ctoken_generate",
        "output_p2ctoken": "output_P2Ctoken_generate",
        "p2ctoken": "output_P2Ctoken_generate",

        "output_p2rtoken_generate": "output_P2Rtoken_generate",
        "p2rtoken_generate": "output_P2Rtoken_generate",
        "output_p2rtoken": "output_P2Rtoken_generate",
        "p2rtoken": "output_P2Rtoken_generate",

        "output_r2atoken_generate": "output_R2Atoken_generate",
        "r2atoken_generate": "output_R2Atoken_generate",
        "output_r2atoken": "output_R2Atoken_generate",
        "r2atoken": "output_R2Atoken_generate",

        "output_r2ctoken_generate": "output_R2Ctoken_generate",
        "r2ctoken_generate": "output_R2Ctoken_generate",
        "output_r2ctoken": "output_R2Ctoken_generate",
        "r2ctoken": "output_R2Ctoken_generate",

        "output_r2ptoken_generate": "output_R2Ptoken_generate",
        "r2ptoken_generate": "output_R2Ptoken_generate",
        "output_r2ptoken": "output_R2Ptoken_generate",
        "r2ptoken": "output_R2Ptoken_generate",

                                                            
        "output_officehome-crconcept_token20_generate": "output_OfficeHome-CRconcept_token20_generate",
        "officehome-crconcept_token20_generate": "output_OfficeHome-CRconcept_token20_generate",
        "crconcept_token20_generate": "output_OfficeHome-CRconcept_token20_generate",
                                                                    
        "output_officehome-crconcept_token5_generate": "output_OfficeHome-CRconcept_token5_generate",
        "officehome-crconcept_token5_generate": "output_OfficeHome-CRconcept_token5_generate",
        "crconcept_token5_generate": "output_OfficeHome-CRconcept_token5_generate",

        "output_officehome-crconcept_token50_generate": "output_OfficeHome-CRconcept_token50_generate",
        "officehome-crconcept_token50_generate": "output_OfficeHome-CRconcept_token50_generate",
        "crconcept_token50_generate": "output_OfficeHome-CRconcept_token50_generate",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    DOMAIN_DIRS = {
        "Ar": "Art",
        "Cl": "Clipart",
        "Pr": "Product",
        "Rw": "Real_World",
        "output_Ar2Cl": "output_Ar2Cl",
        "output_R2C_generate": "output_R2C_generate",
        "output_R2P_generate": "output_R2P_generate",
        "output_A2C_generate": "output_A2C_generate",
        "output_A2P_generate": "output_A2P_generate",
        "output_A2R_generate": "output_A2R_generate",
        "RealWorldCustom": "Real World",
        "output_C2A_generate": "output_C2A_generate",
        "output_C2P_generate": "output_C2P_generate",
        "output_C2R_generate": "output_C2R_generate",
        "output_P2A_generate": "output_P2A_generate",
        "output_P2C_generate": "output_P2C_generate",
        "output_P2R_generate": "output_P2R_generate",
        "output_R2A_generate": "output_R2A_generate",
                                                                   
        "output_OfficeHome-CRconcept_token20_generate": "output_OfficeHome-CRconcept_token20_generate",
                                                                            
        "output_OfficeHome-CRconcept_token5_generate": "output_OfficeHome-CRconcept_token5_generate",
        "output_OfficeHome-CRconcept_token50_generate": "output_OfficeHome-CRconcept_token50_generate",
    }

    @classmethod
    def _download_entries_for_task(cls, task: str):
        """Only download/check what's needed for the requested domain."""
        picked = []
        for entry in cls.download_list:
            name = entry[0]
            if name == "image_list" or name == task:
                picked.append(entry)
        return picked

    def __init__(
        self,
        root: str,
        task: str,
        download: bool = False,
        subset_per_class: Optional[int] = None,
        subset_seed: int = 0,
        subset_strict: bool = True,
        **kwargs
    ):
                                                                              
        office31_like = any(os.path.isdir(os.path.join(root, d)) for d in ("amazon", "dslr", "webcam"))
        officehome_like = any(os.path.isdir(os.path.join(root, d)) for d in ("Art", "Clipart", "Product", "Real_World", "image_list"))
        if office31_like and not officehome_like:
            raise ValueError(
                f"OfficeHome root looks incorrect (found Office-31 folders under '{root}').\n"
                "Fix: pass the OfficeHomeDataset_10072016 root (it should contain Art/Clipart/Product/Real_World and image_list/)."
            )

                                                   
        norm = self._normalize_task(task)
        assert norm in self.image_list, (
            f"Unknown OfficeHome domain '{task}'. Use one of: Art/Clipart/Product/Real_World or Ar/Cl/Pr/Rw, "
            "plus custom output_Ar2Cl/output_R2C_generate/output_R2P_generate/"
            "output_A2C_generate/output_A2P_generate/output_A2R_generate/"
            "output_C2A_generate/output_C2P_generate/output_C2R_generate/"
            "output_P2A_generate/output_P2C_generate/output_P2R_generate/"
            "output_R2A_generate/Real World."
        )

                                                                               
        self._ensure_image_lists(root)

        data_list_file = os.path.join(root, self.image_list[norm])
        if not os.path.isfile(data_list_file):
            raise FileNotFoundError(
                f"Missing OfficeHome image list file: {data_list_file}\n"
                "Fix: ensure root points to OfficeHomeDataset_10072016 and contains image_list/*.txt "
                "(or the corresponding domain folders so lists can be generated)."
            )

                                                                  
        if subset_per_class is not None:
            try:
                data_list_file = self._build_kshot_list_file(
                    root=root,
                    norm_task=norm,
                    base_list_file=data_list_file,
                    k=int(subset_per_class),
                    seed=int(subset_seed),
                    strict=bool(subset_strict),
                )
            except ValueError as e:
                                                                                                            
                domain = self.DOMAIN_DIRS.get(norm, str(norm))
                self._ensure_image_lists(root, force_domains={domain})
                data_list_file = os.path.join(root, self.image_list[norm])
                data_list_file = self._build_kshot_list_file(
                    root=root,
                    norm_task=norm,
                    base_list_file=data_list_file,
                    k=int(subset_per_class),
                    seed=int(subset_seed),
                    strict=bool(subset_strict),
                )

        missing = self._missing_components(root)
        if download and missing:
                                                                             
            task_dir = self.DOMAIN_DIRS.get(norm)
            for name, archive, url in self.download_list:
                if name not in missing:
                    continue
                if name == "image_list" or (task_dir is not None and name == task_dir):
                    download_data(root, name, archive, url)

        super(OfficeHome, self).__init__(root, OfficeHome.CLASSES, data_list_file=data_list_file, **kwargs)

    @staticmethod
    def _build_kshot_list_file(
        root: str,
        norm_task: str,
        base_list_file: str,
        k: int,
        seed: int = 0,
        strict: bool = True,
    ) -> str:
        """
        Build a derived ImageList file that keeps at most k samples per class.
        Output path: <root>/image_list/<domain>_k{k}_seed{seed}.txt
        """
        if k <= 0:
            raise ValueError(f"subset_per_class must be > 0, got {k}")

        domain = OfficeHome.DOMAIN_DIRS.get(norm_task, str(norm_task))
        safe_domain = domain.replace(" ", "_").replace("-", "_")
        out_file = os.path.join(root, "image_list", f"{safe_domain}_k{k}_seed{seed}.txt")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
                                                                                                    
            try:
                if os.path.getmtime(out_file) >= os.path.getmtime(base_list_file):
                    return out_file
            except Exception:
                return out_file

        per_class: dict[int, list[str]] = {}
        with open(base_list_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    y = int(parts[-1])
                except Exception:
                    continue
                per_class.setdefault(y, []).append(raw if raw.endswith("\n") else (raw + "\n"))

        rng = random.Random(seed)
        picked: list[str] = []
        missing = []
        for y in range(len(OfficeHome.CLASSES)):
            items = per_class.get(y, [])
            if len(items) < k:
                missing.append((y, len(items)))
                if strict:
                    continue
            rng.shuffle(items)
            picked.extend(items[: min(k, len(items))])

        if missing and strict:
            msg = "; ".join([f"class={y} has {n} (<{k})" for y, n in missing[:20]])
            raise ValueError(
                "Not enough images to build k-shot subset for all classes. "
                f"(k={k}, seed={seed}) Missing: {msg}"
            )

                                                         
        rng.shuffle(picked)
        with open(out_file, "w") as f:
            f.writelines(picked)
        return out_file

    @staticmethod
    def _normalize_task(task: str) -> str:
                                                                                                                     
        if os.path.isabs(task) and os.path.isdir(task):
            base = os.path.basename(task.rstrip('/'))
            reverse_std = {"Art": "Ar", "Clipart": "Cl", "Product": "Pr", "Real_World": "Rw"}
            if base in reverse_std:
                return reverse_std[base]
            if base in (
                "output_Ar2Cl",
                "output_R2C_generate",
                "output_R2P_generate",
                "output_A2C_generate",
                "output_A2P_generate",
                "output_A2R_generate",
                "Real World",
                "output_C2A_generate",
                "output_C2P_generate",
                "output_C2R_generate",
                "output_P2A_generate",
                "output_P2C_generate",
                "output_P2R_generate",
                "output_R2A_generate",
                                                                 
                "output_OfficeHome-CRconcept_token20_generate",
                                                                         
                "output_OfficeHome-CRconcept_token5_generate",
                "output_OfficeHome-CRconcept_token50_generate",
            ):
                if base == "Real World":
                    return "RealWorldCustom"
                return base

        if task in (
            "Ar",
            "Cl",
            "Pr",
            "Rw",
            "output_Ar2Cl",
            "output_R2C_generate",
            "output_R2P_generate",
            "output_A2C_generate",
            "output_A2P_generate",
            "output_A2R_generate",
            "RealWorldCustom",
            "output_C2A_generate",
            "output_C2P_generate",
            "output_C2R_generate",
            "output_P2A_generate",
            "output_P2C_generate",
            "output_P2R_generate",
                                                            
            "output_OfficeHome-CRconcept_token20_generate",
                                                                    
            "output_OfficeHome-CRconcept_token5_generate",
            "output_OfficeHome-CRconcept_token50_generate",
        ):
            return task

        key = str(task).strip().lower()
        return OfficeHome.TASK_ALIASES.get(key, task)

    def _missing_components(self, root: str):
        return [name for name, _, _ in self.download_list if not os.path.exists(os.path.join(root, name))]

    def _ensure_image_lists(self, root: str, force_domains: set[str] | None = None) -> None:
                                                                     
        img_list_dir = os.path.join(root, "image_list")
        os.makedirs(img_list_dir, exist_ok=True)

                                                                     
        def _norm(s: str) -> str:
            return s.strip().lower().replace(' ', '_').replace('-', '_')

        canonical_map = {_norm(c): c for c in self.CLASSES}
        class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        for code, domain in self.DOMAIN_DIRS.items():
            if force_domains is not None and domain not in force_domains:
                continue

            list_path = os.path.join(img_list_dir, f"{domain}.txt")
            domain_dir = os.path.join(root, domain)

                                                                            
            if (force_domains is not None or not os.path.isfile(list_path)) and os.path.isdir(domain_dir):
                lines = []
                                                  
                for cls in sorted(os.listdir(domain_dir)):
                    cls_dir = os.path.join(domain_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                                                                   
                    canon = canonical_map.get(_norm(cls))
                    if canon is None:
                        continue                             
                    label = class_to_idx[canon]
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            rel_path = os.path.join(domain, cls, fname)
                            lines.append(f"{rel_path} {label}\n")
                                                                                  
                with open(list_path, "w") as f:
                    f.writelines(lines)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
