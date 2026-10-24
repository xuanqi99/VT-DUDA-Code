                                                             
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
import glob
              
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

_THIS_DIR = osp.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PARENT_DIR = osp.abspath(osp.join(_THIS_DIR, '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, '..', '..'))
_DEFAULT_VISDA_ROOT = osp.join(_PROJECT_ROOT, 'dataset', 'visda2017')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature
from common.utils.sam import SAM

                                                            
from common.vision.datasets.visda2017 import VisDA2017
from common.vision.datasets.imagelist import ImageList

import utils
from generated_data import add_list_arguments, get_list_datasets, list_inputs_requested


                                                                                     
import hashlib

def _fingerprint_lines(lines, k: int = 200) -> str:
    h = hashlib.md5()
    for s in lines[:k]:
        h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


                                                                           
def _ensure_visda_imagelist(root_dir: str, classes, out_file: str,
                            exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp"),
                            absolute_paths: bool = False) -> str:
    """Create `out_file` in ImageList format: `path label` per line."""
    if osp.isfile(out_file) and osp.getsize(out_file) > 0:
        return out_file

    root_dir = osp.abspath(osp.expanduser(root_dir))
    out_file = osp.abspath(osp.expanduser(out_file))
    os.makedirs(osp.dirname(out_file), exist_ok=True)

    missing_class_dirs = []
    per_class_counts = {cls: 0 for cls in classes}
    lines = []

    for label, cls in enumerate(classes):
        cls_dir = osp.join(root_dir, cls)
        if not osp.isdir(cls_dir):
            missing_class_dirs.append(cls_dir)
            continue

        files = sorted(glob.glob(osp.join(cls_dir, "**", "*"), recursive=True))
        for fp in files:
            if not osp.isfile(fp):
                continue
            if osp.splitext(fp)[1].lower() not in exts:
                continue
            per_class_counts[cls] += 1
            if absolute_paths:
                path_str = osp.abspath(fp).replace("\\", "/")
            else:
                path_str = osp.relpath(fp, root_dir).replace("\\", "/")
            lines.append(f"{path_str} {label}\n")

                                                                                            
    empty_classes = [c for c, n in per_class_counts.items() if n == 0]
    if missing_class_dirs or empty_classes:
        msg = [f"Invalid VisDA source at '{root_dir}': class folders/images are incomplete."]
        if missing_class_dirs:
            msg.append("Missing class dirs (examples):")
            msg.extend(missing_class_dirs[:12])
        if empty_classes:
            msg.append(f"Classes with 0 images: {empty_classes}")
        msg.append("Expected 12 class subfolders exactly matching VisDA2017.CLASSES.")
        raise FileNotFoundError("\n".join(msg))

    if not lines:
        raise FileNotFoundError(f"Failed to build image_list.txt under '{root_dir}'. No images found.")

    with open(out_file, "w") as f:
        f.writelines(lines)
    print(f"[info] Generated ImageList: {out_file} (num_images={len(lines)})")
    return out_file


def _get_visda2017_datasets(args, train_transform, val_transform):
    """
    VisDA2017:
      - source: generated images at `<root>/output_visda2017concept_generate/` (must contain class subdirs)
      - target: official labeled validation/ (Real). `test/` is typically unlabeled, not used here.
    """
    visda_root = osp.abspath(osp.expanduser(args.root))

                                                                                   
    default_source_name = "output_visda2017concept_generate"
    src_hint = (args.source[0] if args.source else default_source_name)

                         
    candidate = src_hint
    if not osp.isabs(candidate):
        candidate = osp.join(visda_root, candidate)
    candidate = osp.abspath(candidate)

    def _looks_like_valid_visda_source_dir(p: str) -> bool:
        if not osp.isdir(p):
            return False
                                        
        return all(osp.isdir(osp.join(p, c)) for c in VisDA2017.CLASSES)

    if not _looks_like_valid_visda_source_dir(candidate):
        fallback = osp.join(visda_root, default_source_name)
        if _looks_like_valid_visda_source_dir(fallback):
            print(f"[warn] VisDA source='{src_hint}' resolved to '{candidate}' but it's invalid/missing classes. "
                  f"Falling back to '{fallback}'. Pass -s <dir> to override.")
            source_dir = fallback
        else:
                                          
            raise FileNotFoundError(
                f"VisDA source dir invalid: '{candidate}' (from -s/--source '{src_hint}'). "
                f"Also fallback '{fallback}' is invalid. "
                f"Expected 12 class subfolders: {VisDA2017.CLASSES}"
            )
    else:
        source_dir = candidate

                                                                          
    if not osp.isdir(osp.join(visda_root, "validation")):
        raise FileNotFoundError(
            f"VisDA2017 target split 'validation' not found under root='{visda_root}'. "
            f"Found dirs: {[d for d in ['train','validation','test'] if osp.isdir(osp.join(visda_root,d))]}"
        )

                                  
    source_list_file = osp.join(source_dir, "image_list.txt")

                                                                                      
    _ensure_visda_imagelist(source_dir, VisDA2017.CLASSES, source_list_file, absolute_paths=False)

                                                                                                
    try:
        with open(source_list_file, "r") as f:
            _lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"[debug] VisDA source_dir='{source_dir}'")
        print(f"[debug] VisDA source_list_file='{source_list_file}', num_lines={len(_lines)}, md5@200={_fingerprint_lines(_lines, 200)}")
        print(f"[debug] VisDA source_list head: {_lines[:3]}")
    except Exception as e:
        print(f"[warn] Failed to read/inspect source_list_file='{source_list_file}': {e}")

    train_source_dataset = ImageList(
        root=source_dir,
        classes=VisDA2017.CLASSES,
        data_list_file=source_list_file,
        transform=train_transform
    )

                                                                        
    train_target_dataset = VisDA2017(
        root=visda_root, task="Real", download=False, transform=train_transform
    )
    val_dataset = VisDA2017(
        root=visda_root, task="Real", download=False, transform=val_transform
    )
    test_dataset = val_dataset

                                                                 
    try:
        print(f"[debug] VisDA target(train)={train_target_dataset}")
        print(f"[debug] VisDA target(val)={val_dataset}")
    except Exception:
        pass

    num_classes = len(VisDA2017.CLASSES)
    class_names = VisDA2017.CLASSES
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def _scale_param_group_lrs_as_multipliers(optim, base_lr: float) -> None:
    """
    Convert per-group lr "multipliers" (e.g. 0.1/1.0) into absolute lrs (multiplier * base_lr),
    but only when it really looks like multipliers.

    This must be idempotent (calling twice should not keep shrinking lrs).
    """
    if optim is None:
        return
    try:
        base_lr = float(base_lr)
    except Exception:
        return
    if base_lr <= 0:
        return

                         
    lrs = []
    for g in optim.param_groups:
        try:
            lrs.append(float(g.get("lr", base_lr)))
        except Exception:
            lrs.append(base_lr)

    if not lrs:
        return

    max_lr = max(lrs)

                
                                               
                                                                                                        
                                                                                                 
    ratio = max_lr / base_lr if base_lr > 0 else 0.0
    looks_like_multipliers = (0.01 <= max_lr <= 10.0) and (ratio >= 5.0)

    if not looks_like_multipliers:
        return

    for g in optim.param_groups:
        lr_val = float(g.get("lr", base_lr))
                                                                                   
        if 0.0 <= lr_val <= 10.0:
            g["lr"] = lr_val * base_lr


def _load_checkpoint_into_model(model: nn.Module, ckpt_path: str, *, args: argparse.Namespace, strict: bool = True):
    """Load checkpoint with backward compatibility + meta validation."""
    obj = torch.load(ckpt_path, map_location="cpu")
    meta = None
    state = obj
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta = obj.get("meta", None)

                                                                                          
    if isinstance(meta, dict):
        arch0 = meta.get("arch", None)
        nc0 = meta.get("num_classes", None)
        arch1 = getattr(args, "arch", None)
        nc1 = getattr(args, "num_classes", None)
        if (arch0 is not None and arch1 is not None and str(arch0) != str(arch1)) or\
           (nc0 is not None and nc1 is not None and int(nc0) != int(nc1)):
            raise RuntimeError(
                "Checkpoint/model mismatch:\n"
                f"  ckpt_path={ckpt_path}\n"
                f"  ckpt.meta.arch={arch0}, ckpt.meta.num_classes={nc0}\n"
                f"  current.arch={arch1}, current.num_classes={nc1}\n"
                "This usually happens when you reuse the same --log directory across different runs.\n"
                "Fix: use a fresh log dir (this script now supports timestamped run dirs), or load a matching checkpoint."
            )

    model.load_state_dict(state, strict=strict)
    return meta

def _save_checkpoint(model: nn.Module, ckpt_path: str, *, args: argparse.Namespace, meta_extra: dict | None = None):
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arch": getattr(args, "arch", None),
        "num_classes": int(getattr(args, "num_classes", -1)) if getattr(args, "num_classes", None) is not None else None,
        "data": getattr(args, "data", None),
        "root": getattr(args, "root", None),
        "source": getattr(args, "source", None),
        "target": getattr(args, "target", None),
        "eps": getattr(args, "eps", None),
        "run_id": getattr(args, "run_id", None),
    }
    if meta_extra:
        meta.update(meta_extra)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)


def _confmat_update_(confmat: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
    """In-place confusion-matrix update. confmat shape: [C, C], rows=target, cols=pred."""
    c = confmat.size(0)
    t = targets.view(-1).to(torch.int64)
    p = preds.view(-1).to(torch.int64)
    idx = t * c + p
    binc = torch.bincount(idx, minlength=c * c).reshape(c, c)
    confmat += binc.to(confmat.device)

def _per_class_acc_from_confmat(confmat: torch.Tensor) -> torch.Tensor:
    """Return per-class accuracy (%) as float tensor [C]."""
    confmat = confmat.to(torch.float32)
    correct = torch.diag(confmat)
    total = confmat.sum(dim=1).clamp_min(1.0)
    return 100.0 * correct / total

@torch.no_grad()
def _validate_visda_with_per_class(loader: DataLoader, model: nn.Module, *, device: torch.device, class_names) -> tuple[float, list[float], float]:
    """Return (acc1, per_class_acc_list, mean_per_class_acc)."""
    model.eval()
    num_classes = len(class_names)
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")

    correct = 0
    total = 0

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)

        out = model(x)

                                                                       
        logits = out[0] if isinstance(out, (tuple, list)) else out

                                                                           
        if isinstance(logits, torch.Tensor) and logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
            raise RuntimeError(f"Unexpected logits shape from model: {type(logits)} with shape={getattr(logits, 'shape', None)}")

        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

        _confmat_update_(confmat, y.detach().cpu(), pred.detach().cpu())

    acc1 = 100.0 * float(correct) / float(max(total, 1))
    per_class = _per_class_acc_from_confmat(confmat).cpu().tolist()
    mean_pc = float(np.mean([v for i, v in enumerate(per_class) if confmat[i].sum().item() > 0])) if num_classes > 0 else 0.0
    return acc1, per_class, mean_pc


def main(args: argparse.Namespace, eps=0.):
                                                                    
    if not hasattr(args, "norm_std") or not hasattr(args, "norm_mean"):
        raise AttributeError("args must have norm_mean and norm_std. Check argparse / callsites.")

    logger = CompleteLogger(args.log, args.phase)
    print(args)

                          
                                                       
                                   
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    device = args.device

                                                                                       
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except Exception:
        available_cpus = os.cpu_count() or 1
    if args.workers > available_cpus:
        print(f"[warn] Reducing --workers from {args.workers} to {available_cpus} (available CPUs).")
        args.workers = available_cpus

                       
    train_transform = utils.get_train_transform(
        args.train_resizing,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,                          
    )
    val_transform = utils.get_val_transform(
        args.val_resizing,
        resize_size=args.resize_size,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,                          
    )
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

                                                                        
    if list_inputs_requested(args):
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            get_list_datasets(args, train_transform, val_transform)
    elif args.data == "VisDA2017":
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names =\
            _get_visda2017_datasets(args, train_transform, val_transform)
    else:
        if args.dataset_ratio == -1:
            train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names =\
                utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
        else:
            train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names =\
                utils.get_ratio_dataset(args.data, args.root, args.source,
                                        args.target, train_transform, val_transform, args.dataset_ratio)

                                                                    
    args.num_classes = int(num_classes)

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

                           
    print(f"[debug] #train_source={len(train_source_dataset)}, "
          f"#train_target={len(train_target_dataset)}, "
          f"#val={len(val_dataset)}, #test={len(test_dataset)}, "
          f"num_classes={num_classes}")

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

                  
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
                     
    pool_layer = nn.Identity() if args.no_pool else None
                       
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier.features_dim

                                                                       
    if not args.mcc_only:
        if args.randomized:
            domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        else:
            domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
    else:
        domain_discri = None

                                       
    base_optimizer = torch.optim.SGD
    if not args.mcc_only:
        ad_optimizer = SGD(domain_discri.get_parameters(), args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)
    else:
        ad_optimizer = None

    optimizer = SAM(classifier.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

                                                                                          
    _scale_param_group_lrs_as_multipliers(optimizer.base_optimizer, args.lr)
    _scale_param_group_lrs_as_multipliers(ad_optimizer, args.lr)

                                                             
                                                                                                                
    def _lr_mult(step: int) -> float:
        return (1. + args.lr_gamma * float(step)) ** (-args.lr_decay)

    _sched_opt = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
    lr_scheduler = LambdaLR(_sched_opt, lr_lambda=_lr_mult)

    if not args.mcc_only:
        lr_scheduler_ad = LambdaLR(ad_optimizer, lr_lambda=_lr_mult)
    else:
        lr_scheduler_ad = None

                          
    if not args.mcc_only:
        domain_adv = ConditionalDomainAdversarialLoss(
            domain_discri, entropy_conditioning=args.entropy,
            num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
            randomized_dim=args.randomized_dim, eps=eps
        ).to(device)
    else:
        domain_adv = None

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

                                     
    if args.phase != 'train':
                                                                                 
        cand = [
            logger.get_checkpoint_path(f'best_with_eps{getattr(args, "eps", eps)}'),
            logger.get_checkpoint_path('best'),
            logger.get_checkpoint_path('latest'),
        ]
        ckpt_path = next((p for p in cand if osp.isfile(p)), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found. Tried: {cand}")
        _load_checkpoint_into_model(classifier, ckpt_path, args=args, strict=True)

                                      
    if args.phase == 'finetune':
        path = args.weight_path
        print(f"[INFORMATION] Using the weights stored at {args.weight_path}")
        classifier.load_state_dict(torch.load(path))

                        
    if args.phase == 'analysis':
        path = args.weight_path
        print(f"[INFORMATION] Using the weights stored at {args.weight_path}")
        classifier.load_state_dict(torch.load(path))
                                            
        feature_extractor = nn.Sequential(
            classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(
            train_source_loader, feature_extractor, device)
        print(source_feature.size())
        target_feature = collect_feature(
            train_target_loader, feature_extractor, device)
        A_distance = a_distance.calculate(
            source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return 1


    if args.phase == 'test':
        if args.data == "VisDA2017":
            acc1, per_cls, mean_pc = _validate_visda_with_per_class(
                test_loader, classifier, device=device, class_names=getattr(args, "class_names", VisDA2017.CLASSES)
            )
            print(f" * Acc@1 {acc1:.3f}  Mean-per-class {mean_pc:.3f}")
            for name, v in zip(getattr(args, "class_names", VisDA2017.CLASSES), per_cls):
                print(f"   - {name}: {v:.3f}")
            return
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    all_accuracy = []

                    
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])

        [acc, loss] = train(train_source_iter, train_target_iter, classifier, domain_adv, mcc_loss, optimizer,
                            ad_optimizer,
                            lr_scheduler, lr_scheduler_ad, epoch, args)

                                    
        if args.data == "VisDA2017":
            acc1, per_cls, mean_pc = _validate_visda_with_per_class(
                val_loader, classifier, device=device, class_names=getattr(args, "class_names", VisDA2017.CLASSES)
            )
            print(f"[debug] Epoch {epoch} val Acc@1 = {acc1:.3f}  Mean-per-class = {mean_pc:.3f}")
            for name, v in zip(getattr(args, "class_names", VisDA2017.CLASSES), per_cls):
                print(f"[per-class] {name}: {v:.3f}")
        else:
            acc1 = utils.validate(val_loader, classifier, args, device)
            print(f"[debug] Epoch {epoch} val Acc@1 = {acc1:.3f}")

        all_accuracy.append([acc1, loss])

                                                 
        _save_checkpoint(classifier, logger.get_checkpoint_path('latest'), args=args, meta_extra={"epoch": epoch, "val_acc1": float(acc1)})

        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path(f'best_with_eps{eps}'))
        best_acc1 = max(acc1, best_acc1)

    all_accuracy = np.array(all_accuracy)
    np.save(os.path.join(args.log, "acc_and_loss"), all_accuracy)
    print("best_acc1 = {:3.2f}".format(best_acc1))

                          
    best_path = logger.get_checkpoint_path(f'best_with_eps{eps}')
    if not osp.isfile(best_path):
        best_path = logger.get_checkpoint_path('best')

                                                                                  
    _load_checkpoint_into_model(classifier, best_path, args=args, strict=True)

    if args.data == "VisDA2017":
        acc1, per_cls, mean_pc = _validate_visda_with_per_class(
            test_loader, classifier, device=device, class_names=getattr(args, "class_names", VisDA2017.CLASSES)
        )
        print(f"test_acc1 = {acc1:.2f}  mean_per_class = {mean_pc:.2f}")
        for name, v in zip(getattr(args, "class_names", VisDA2017.CLASSES), per_cls):
            print(f"[per-class-test] {name}: {v:.3f}")
        return acc1

    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.2f}".format(acc1))
    return acc1


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, mcc, optimizer, ad_optimizer,
          lr_scheduler: LambdaLR, lr_scheduler_ad, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    if not args.mcc_only:
        trans_losses = AverageMeter('Trans Loss', ':3.2f')
        domain_accs = AverageMeter('Domain Acc', ':3.1f')
        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs],
            prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            args.iters_per_epoch,
            [batch_time, data_time, losses, cls_accs],
            prefix="Epoch: [{}]".format(epoch))

                          
    model.train()
    if domain_adv is not None:
        domain_adv.train()

    device = args.device
    bad_batch_count = 0
    did_sanity = False

    end = time.time()
    for i in range(args.iters_per_epoch):
                            
        try:
            x_s, labels_s = next(train_source_iter)
        except (OSError, IOError) as e:
            bad_batch_count += 1
            if bad_batch_count <= 10:
                print(f"[warn] Source loader error at iter {i}, epoch {epoch}: {e}")
            continue
        except RuntimeError as e:
            err_msg = str(e)
            if "Caught OSError in DataLoader worker process" in err_msg or "image file" in err_msg:
                bad_batch_count += 1
                if bad_batch_count <= 10:
                    print(f"[warn] Source loader error at iter {i}, epoch {epoch}: {err_msg}")
                continue
            raise

                            
        try:
            x_t, _ = next(train_target_iter)
        except (OSError, IOError) as e:
            bad_batch_count += 1
            if bad_batch_count <= 10:
                print(f"[warn] Target loader error at iter {i}, epoch {epoch}: {e}")
            continue
        except RuntimeError as e:
            err_msg = str(e)
            if "Caught OSError in DataLoader worker process" in err_msg or "image file" in err_msg:
                bad_batch_count += 1
                if bad_batch_count <= 10:
                    print(f"[warn] Target loader error at iter {i}, epoch {epoch}: {err_msg}")
                continue
            raise

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        if (not did_sanity) and epoch == 0 and i == 0:
            did_sanity = True
            with torch.no_grad():
                                                                             
                num_classes = int(getattr(args, "num_classes", -1))
                if num_classes <= 0:
                                                   
                    _x = torch.cat((x_s, x_t), dim=0)
                    _y, _ = model(_x)
                    num_classes = int(_y.size(1))
                    args.num_classes = num_classes

                mn = int(labels_s.min().item())
                mx = int(labels_s.max().item())
                print(f"[sanity] labels_s range: [{mn}, {mx}]  (expected within [0, {num_classes - 1}])")
                if mn < 0 or mx >= num_classes:
                    raise ValueError(
                        f"labels_s out of range: [{mn}, {mx}] vs num_classes={num_classes}. "
                        f"Check ImageList/VisDA label mapping."
                    )
                bc = torch.bincount(labels_s.detach().cpu(), minlength=num_classes)
                print(f"[sanity] labels_s bincount (first batch): {bc.tolist()}")

                                   
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        if ad_optimizer is not None:
            ad_optimizer.zero_grad()

                        
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

                        
        cls_loss = F.cross_entropy(y_s, labels_s)
        if args.mcc_only:
            loss = cls_loss + mcc(y_t)
        else:
            mcc_loss_value = mcc(y_t)
            loss = cls_loss + mcc_loss_value
        loss.backward()
        optimizer.first_step(zero_grad=True)

                         
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        if args.mcc_only:
            transfer_loss = mcc(y_t)
            loss = cls_loss + transfer_loss
        else:
            transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        if not args.mcc_only:
            domain_accs.update(domain_acc, x_s.size(0))
            trans_losses.update(transfer_loss.item(), x_s.size(0))

        loss.backward()
        if ad_optimizer is not None:
            ad_optimizer.step()
        optimizer.second_step(zero_grad=True)

        lr_scheduler.step()
        if lr_scheduler_ad is not None:
            lr_scheduler_ad.step()

                              
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

                                     
    if bad_batch_count > 0:
        print(f"[info] Epoch {epoch}: skipped {bad_batch_count} batches due to data loading errors.")
                        
    return [cls_accs.avg.cpu().numpy(), losses.avg]


                                                               
EPS_SWEEP_DEFAULT = [0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    add_list_arguments(parser)

                        
    parser.add_argument('--save-acc', action='store_true',
                        help='whether to save acc')
                                          
    parser.add_argument(
        'root',
        metavar='DIR',
        nargs='?',
        default=_DEFAULT_VISDA_ROOT,
        help='root path of dataset'
    )

                                                                                           
    _dataset_choices = list(utils.get_dataset_names())
    if "VisDA2017" not in _dataset_choices:
        _dataset_choices.append("VisDA2017")

    parser.add_argument(
        '-d', '--data',
        metavar='DATA',
        default='VisDA2017',
        choices=_dataset_choices,
        help='dataset: ' + ' | '.join(_dataset_choices) +
             ' (default: VisDA2017)'
    )
                                                  
    parser.add_argument(
        '-s', '--source',
        help='source domain(s)',
        nargs='+',
        default=['A']
    )
    parser.add_argument(
        '-t', '--target',
        help='target domain(s)',
        nargs='+',
        default=['W']
    )
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
                      
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet101)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--eps', default=1.0, type=float,
                        help='hyper-parameter for environemnt label smoothing.')
    parser.add_argument('--dataset-ratio', default=-1, type=float,
                        help='The ratio between dataset1 and dataset2, it will crop the last dataset in the given '
                             'datasets rate')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
                         
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,     
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--weight_path", type=str, default='cdan',
                        help="Path to the saved weights")
    parser.add_argument("--phase", type=str, default='train',
                        choices=['train', 'test', 'analysis', 'finetune', 'analysis_distance'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--log_results', action='store_true',
                        help="To log results in wandb")
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID")
    parser.add_argument('--log_name', type=str,
                        default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    parser.add_argument('--mcc-only', action='store_true',
                        help='use only classification + MCC loss (disable CDAN/domain adversarial).')
                                                                     
                                                           
                                                                                                        
                                                                                                                   
                                                                     
    parser.add_argument('maybe_path', nargs='?', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase in {'analysis', 'analysis_distance'}:
        parser.error("Feature analysis requires tsne/a_distance modules absent from this source release.")
                                                                     
    if isinstance(args.root, str) and args.root.strip().lower() == 'root' and args.maybe_path:
        args.root = args.maybe_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                                   
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    print(f"[debug] CUDA_VISIBLE_DEVICES='{visible}', cuda.is_available={cuda_available}, device_count={cuda_count}")

    if cuda_available and cuda_count > 0 and visible.strip() != "":
        try:
                                       
            torch.cuda.set_device(0)
            device = torch.device("cuda")
            print(f"[info] Using CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[warn] Failed to set CUDA device (reason: {e}). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        if not cuda_available:
            print("[warn] torch.cuda.is_available() is False. Check your PyTorch build and NVIDIA driver.")
        elif cuda_count == 0:
            print("[warn] No visible CUDA devices. Check CUDA_VISIBLE_DEVICES or GPU allocation.")
        device = torch.device("cpu")

    args.device = device

                                                                                                                                                 
    def _path_to_domain(entries, root):
        out = []
        for e in entries or []:
            if os.path.isabs(e) and os.path.isdir(e):
                base = os.path.basename(e.rstrip('/'))
                out.append(base)                                                            
            else:
                out.append(e)
        return out
    if args.source:
        args.source = _path_to_domain(args.source, args.root)
    if args.target:
        args.target = _path_to_domain(args.target, args.root)

                                         
    print(f"[info] Using domains as-is: source={args.source}, target={args.target}")

                                                                               
                                                                            
    if args.phase == 'train' and isinstance(args.log, str) and args.log.strip() and not osp.isabs(args.log):
        args.run_id = time.strftime("%Y%m%d-%H%M%S")
        args.log = osp.join(args.log, args.run_id)
        os.makedirs(args.log, exist_ok=True)
        print(f"[info] Run dir: {args.log}")
    else:
        args.run_id = None

    def _eps_dir_name(value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return "eps" + text.replace(".", "p").replace("-", "m")

    if args.eps >= 1:
                                                                                             
        epses = EPS_SWEEP_DEFAULT
        acces = np.array([0. for i in epses])
        base_log = args.log
        base_eps = args.eps
        for i in range(len(epses)):
            eps_value = float(epses[i])
            args.eps = eps_value
            args.log = osp.join(base_log, _eps_dir_name(eps_value))
            os.makedirs(args.log, exist_ok=True)
            print(f"[info] eps run dir: {args.log}")
            print('eps as these epoch is', eps_value)
            acces[i] = main(args, eps=eps_value)
        args.log = base_log
        args.eps = base_eps
        idx = acces.argmax()
        print(f'the best result is {acces[idx]} with eps {epses[idx]}')
        print(f'when eps=0, the result is', acces[0])
    else:
        print('eps as these epoch is', args.eps)
        main(args, eps=args.eps)
        exit()
