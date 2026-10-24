                                                             
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
              
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
from generated_data import add_list_arguments, get_list_datasets, list_inputs_requested
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature
from common.utils.sam import SAM

import utils


def _normalize_office31_domain(d: str) -> str:
    """Normalize common Office31 domain aliases (case-insensitive)."""
    if not isinstance(d, str):
        return d
    dd = d.strip()
    low = dd.lower()
    mapping = {
        "a": "amazon",
        "amazon": "amazon",
        "w": "webcam",
        "webcam": "webcam",
        "d": "dslr",
        "dslr": "dslr",
    }
    return mapping.get(low, dd)

def _preflight_validate_args(args: argparse.Namespace) -> None:
    """
    Fail fast with a readable error if --data and domains/root look incompatible.
    Keeps behavior conservative: do not silently switch datasets.
    """
    data = (args.data or "").strip()
    src = list(args.source or [])
    tgt = list(args.target or [])
    domains = src + tgt

                                                       
    looks_like_office31 = any(
        (isinstance(d, str) and (
            d.lower() in {"amazon", "dslr", "webcam", "a", "d", "w"} or "office31" in d.lower()
        ))
        for d in domains
    )
    looks_like_officehome = any(
        isinstance(d, str) and d.lower() in {"art", "clipart", "product", "real_world", "real world", "ar", "cl", "pr", "rw"}
        for d in domains
    )

    root = getattr(args, "root", "") or ""
    root_low = root.lower()

                                                               
    office31_markers = [osp.join(root, x) for x in ("amazon", "dslr", "webcam")]
    officehome_markers = [osp.join(root, x) for x in ("Art", "Clipart", "Product", "Real_World", "image_list")]
    looks_like_root_office31 = any(osp.isdir(p) for p in office31_markers)
    looks_like_root_officehome = any(osp.isdir(p) for p in officehome_markers)

                                                   
    suggested_officehome = osp.join(_PROJECT_ROOT, "dataset", "OfficeHomeDataset_10072016")
    has_suggested_officehome = osp.isfile(osp.join(suggested_officehome, "image_list", "Art.txt"))

    if data.lower() == "officehome" and looks_like_root_office31 and not looks_like_root_officehome:
        hint = suggested_officehome if has_suggested_officehome else "<path-to-OfficeHomeDataset_10072016>"
        raise SystemExit(
            "Argument/root mismatch detected (OfficeHome vs Office-31 root):\n"
            f"  --data={args.data}\n"
            f"  --root={args.root}\n"
            f"  --source={args.source}\n"
            f"  --target={args.target}\n\n"
            "Your --root directory looks like Office-31 (has amazon/dslr/webcam), but --data is OfficeHome.\n"
            f"Fix: pass --root {hint} (and keep --data OfficeHome)."
        )

    if data.lower() == "office31" and looks_like_root_officehome and not looks_like_root_office31:
        raise SystemExit(
            "Argument/root mismatch detected (Office31 vs OfficeHome root):\n"
            f"  --data={args.data}\n"
            f"  --root={args.root}\n"
            f"  --source={args.source}\n"
            f"  --target={args.target}\n\n"
            "Your --root directory looks like OfficeHome (has Art/Clipart/Product/Real_World), but --data is Office31.\n"
            "Fix: pass --root <path-to-Office-31> (and keep --data Office31)."
        )

                                               
    if data.lower() == "officehome" and "office-31" in root_low:
        print(f"[warn] --data=OfficeHome but --root looks like Office-31: '{root}'")
    if data.lower() == "office31" and "officehome" in root_low:
        print(f"[warn] --data=Office31 but --root looks like OfficeHome: '{root}'")

                                            
    if data.lower() == "officehome" and looks_like_office31:
        raise SystemExit(
            "Argument mismatch detected:\n"
            f"  --data={args.data}\n"
            f"  --root={args.root}\n"
            f"  --source={args.source}\n"
            f"  --target={args.target}\n\n"
            "Your domains look like Office31 (e.g., amazon/webcam/dslr), but --data is OfficeHome.\n"
            "Fix: pass '--data Office31' (and ensure --root points to the Office-31 dataset root)."
        )

    if data.lower() == "office31" and looks_like_officehome:
        raise SystemExit(
            "Argument mismatch detected:\n"
            f"  --data={args.data}\n"
            f"  --root={args.root}\n"
            f"  --source={args.source}\n"
            f"  --target={args.target}\n\n"
            "Your domains look like OfficeHome (Art/Clipart/Product/Real_World), but --data is Office31.\n"
            "Fix: pass '--data OfficeHome' (and ensure --root points to the OfficeHome dataset root)."
        )

def _scale_param_group_lrs_as_multipliers(optim, base_lr: float) -> None:
    """If param-group lrs look like multipliers (e.g. 0.1/1.0), convert them to absolute lrs once."""
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


def main(args: argparse.Namespace, eps=0.):
                                                                    
    if not hasattr(args, "norm_std") or not hasattr(args, "norm_mean"):
        raise AttributeError("args must have norm_mean and norm_std. Check argparse / callsites.")
    if not (isinstance(args.norm_std, (list, tuple)) and len(args.norm_std) == 3):
        raise ValueError(f"norm_std must be a 3-tuple/list, got: {args.norm_std!r}")
    if not (isinstance(args.norm_mean, (list, tuple)) and len(args.norm_mean) == 3):
        raise ValueError(f"norm_mean must be a 3-tuple/list, got: {args.norm_mean!r}")
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

                       
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    if list_inputs_requested(args):
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
            get_list_datasets(args, train_transform, val_transform)
    elif args.dataset_ratio == -1:
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names =\
            utils.get_dataset(args.data, args.root, args.source,
                              args.target, train_transform, val_transform)

    else:
        train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names =\
            utils.get_ratio_dataset(args.data, args.root, args.source,
                                    args.target, train_transform, val_transform, args.dataset_ratio)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                           
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

                                                                                                                                 
    def _lr_mult(step: int) -> float:
        return (1. + args.lr_gamma * float(step)) ** (-args.lr_decay)

    _sched_opt = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
    _scale_param_group_lrs_as_multipliers(_sched_opt, args.lr)
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
        checkpoint_path = logger.get_checkpoint_path(f'best_with_eps{eps}')
        if not osp.isfile(checkpoint_path):
            checkpoint_path = logger.get_checkpoint_path('best')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                                                           
        classifier.load_state_dict(checkpoint)

                                      
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

                                    
        acc1 = utils.validate(val_loader, classifier, args, device)
                                                      
        print(f"[debug] Epoch {epoch} val Acc@1 = {acc1:.3f}")
        all_accuracy.append([acc1, loss])

                                                 
        torch.save(classifier.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path(f'best_with_eps{eps}'))
        best_acc1 = max(acc1, best_acc1)
    all_accuracy = np.array(all_accuracy)
    np.save(os.path.join(args.log, "acc_and_loss"), all_accuracy)
    print("best_acc1 = {:3.2f}".format(best_acc1))

                          
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path(f'best_with_eps{eps}')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.2f}".format(acc1))
    return acc1


def _confmat_update_(confmat: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor) -> None:
                                                          
    c = confmat.size(0)
    t = targets.view(-1).to(torch.int64).cpu()
    p = preds.view(-1).to(torch.int64).cpu()
    idx = t * c + p
    confmat += torch.bincount(idx, minlength=c * c).reshape(c, c)


def _per_class_acc_from_confmat(confmat: torch.Tensor) -> torch.Tensor:
    confmat = confmat.to(torch.float32)
    correct = torch.diag(confmat)
    total = confmat.sum(dim=1).clamp_min(1.0)
    return 100.0 * correct / total


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

    end = time.time()
    for i in range(args.iters_per_epoch):
                                                             
        try:
            x_s, labels_s = next(train_source_iter)
            x_t, _ = next(train_target_iter)
        except (OSError, IOError) as e:
            bad_batch_count += 1
            if bad_batch_count <= 10:
                print(f"[warn] Caught data loading error at iter {i}, epoch {epoch}: {e}")
                print("[warn] Skipping this batch. Likely due to a corrupted image or unsupported file in imagelist.")
            continue
        except RuntimeError as e:
            err_msg = str(e)
            if "Caught OSError in DataLoader worker process" in err_msg or "image file" in err_msg:
                bad_batch_count += 1
                if bad_batch_count <= 10:
                    print(f"[warn] Caught data loading error at iter {i}, epoch {epoch}: {err_msg}")
                    print("[warn] Skipping this batch. Likely due to a corrupted image or unsupported file in imagelist.")
                continue
            raise
                                                                     

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

                                   
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
        default=None,
        help='root path of dataset (if omitted, a dataset-specific default is used)'
    )

    DEFAULT_ROOT_OFFICE31 = osp.join(_PROJECT_ROOT, "dataset", "Office-31")
    DEFAULT_ROOT_OFFICEHOME = osp.join(_PROJECT_ROOT, "dataset", "OfficeHomeDataset_10072016")

    parser.add_argument(
        '-d', '--data',
        metavar='DATA',
        default='OfficeHome',
        choices=utils.get_dataset_names(),
        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
             ' (default: OfficeHome)'
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
                      
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
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
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
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

                                                                              
    if args.root is None or (isinstance(args.root, str) and args.root.strip() == ""):
        if isinstance(args.data, str) and args.data.lower() == "office31":
            args.root = DEFAULT_ROOT_OFFICE31
        else:
            args.root = DEFAULT_ROOT_OFFICEHOME

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

                                                                                               
    if isinstance(args.data, str) and args.data.lower() == "office31":
        args.source = [_normalize_office31_domain(d) for d in (args.source or [])]
        args.target = [_normalize_office31_domain(d) for d in (args.target or [])]

    print(f"[info] Using domains as-is: source={args.source}, target={args.target}")

                                                                           
    if args.phase == 'train' and isinstance(args.log, str) and args.log.strip() and not osp.isabs(args.log):
        run_id = time.strftime("%Y%m%d-%H%M%S")
        args.log = osp.join(args.log, run_id)
        os.makedirs(args.log, exist_ok=True)
        print(f"[info] Run dir: {args.log}")

    def _eps_dir_name(v: float) -> str:
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return "eps" + s.replace(".", "p").replace("-", "m")

    if args.eps >= 1:
        epses = [0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]
        acces = np.array([0. for i in epses])

        base_log = args.log
        base_eps = args.eps
        for i in range(len(epses)):
            eps_val = float(epses[i])
            args.eps = eps_val
            args.log = osp.join(base_log, _eps_dir_name(eps_val))
            os.makedirs(args.log, exist_ok=True)
            print(f"[info] eps run dir: {args.log}")
            print('eps as these epoch is', eps_val)
            acces[i] = main(args, eps=eps_val)

        args.log = base_log
        args.eps = base_eps
        idx = acces.argmax()
        print(f'the best result is {acces[idx]} with eps {epses[idx]}')
        print(f'when eps=0, the result is', acces[0])
    else:
        print('eps as these epoch is', args.eps)
        main(args, eps=args.eps)
        exit()
