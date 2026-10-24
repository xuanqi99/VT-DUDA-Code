from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31(ImageList):
    """Office31 Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt",
        "A2D": "image_list/AtoD.txt",
        "A2W": "image_list/AtoW.txt",
        "D2A": "image_list/DtoA.txt",
        "D2W": "image_list/DtoW.txt",
        "W2A": "image_list/WtoA.txt",
        "W2D": "image_list/WtoD.txt",
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
                                                                        
                                                                                 
        is_std_task = task in self.image_list

                                                                 
                                                                               
        true_root = root
        parent_root = os.path.dirname(root.rstrip("/"))
        image_list_in_parent = os.path.join(parent_root, "image_list")
        if not os.path.isdir(os.path.join(root, "image_list")) and os.path.isdir(image_list_in_parent):
            true_root = parent_root

                   
                                                 
                                              
        if is_std_task:
            data_list_file = os.path.join(true_root, self.image_list[task])
        else:
            data_list_file = os.path.join(true_root, "image_list", f"{task}.txt")

                                       
        download = False

                                                             
        def _ensure_list_file_exists():
            image_list_dir = os.path.dirname(data_list_file)
            os.makedirs(image_list_dir, exist_ok=True)

                        
            if os.path.exists(data_list_file):
                return

            if is_std_task and task in ("A", "D", "W"):
                                                          
                domain_map = {"A": "amazon", "D": "dslr", "W": "webcam"}
                domain = domain_map[task]
                domain_dir = os.path.join(true_root, domain)
                if not os.path.isdir(domain_dir):
                    raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

                classes = self.CLASSES
                class_to_idx = {c: i for i, c in enumerate(classes)}

                lines = []
                for cls_name in sorted(os.listdir(domain_dir)):
                    cls_path = os.path.join(domain_dir, cls_name)
                    if not os.path.isdir(cls_path):
                        continue
                    if cls_name in class_to_idx:
                        idx = class_to_idx[cls_name]
                    else:
                        norm = cls_name.replace("-", "_")
                        if norm in class_to_idx:
                            idx = class_to_idx[norm]
                        else:
                            continue
                    for root_dir, _, files in os.walk(cls_path):
                        for fn in files:
                            ext = os.path.splitext(fn)[1].lower()
                            if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                                continue
                            abs_path = os.path.join(root_dir, fn)
                            rel_path = os.path.relpath(abs_path, true_root).replace("\\", "/")
                            lines.append(f"{rel_path} {idx}")

                if not lines:
                    raise RuntimeError(f"No images found under {domain_dir} to build {data_list_file}")
                with open(data_list_file, "w") as f:
                    f.write("\n".join(lines))
            else:
                                                                          
                domain_dir = os.path.join(true_root, task)
                if not os.path.isdir(domain_dir):
                    raise FileNotFoundError(
                        f"Custom task '{task}' expects data directory: {domain_dir}"
                    )

                classes = self.CLASSES
                class_to_idx = {c: i for i, c in enumerate(classes)}

                lines = []

                                                  
                                                   
                                      
                for root_dir, _, files in os.walk(domain_dir):
                    cls_name = os.path.basename(root_dir)

                    if cls_name in class_to_idx:
                        idx = class_to_idx[cls_name]
                    else:
                        norm = cls_name.replace("-", "_")
                        if norm in class_to_idx:
                            idx = class_to_idx[norm]
                        else:
                                              
                            continue

                    for fn in files:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                            continue
                        abs_path = os.path.join(root_dir, fn)
                        rel_path = os.path.relpath(abs_path, true_root).replace("\\", "/")
                        lines.append(f"{rel_path} {idx}")

                if not lines:
                    raise RuntimeError(
                        f"No images found under custom task directory {domain_dir} "
                        f"to build {data_list_file}. "
                        f"Expected some sub-folders (at any depth) to be named as one of CLASSES: "
                        f"{classes}."
                    )
                with open(data_list_file, "w") as f:
                    f.write("\n".join(lines))

                            
            if os.path.exists(data_list_file):
                print(f"[debug][Office31] task='{task}' using list file: {data_list_file}")
            else:
                print(f"[warn][Office31] task='{task}' list file still missing: {data_list_file}")
                            

                                       
        if is_std_task:
            for file_name, *_ in self.download_list:
                if file_name == "image_list":
                    continue
                                                                  
                try:
                    check_exits(true_root, file_name)
                except Exception:
                                                 
                    pass

                                   
        _ensure_list_file_exists()

                                
        super(Office31, self).__init__(true_root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
                                         
        return list(cls.image_list.keys())