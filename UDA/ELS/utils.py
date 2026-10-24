import sys
import os.path as osp
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
                    
import os
_THIS_DIR = osp.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PARENT_DIR = osp.abspath(osp.join(_THIS_DIR, '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()




def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
                                               
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
                                               
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
                                                                           
            backbone.out_features = 768
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
                                                    

    return backbone




def get_dataset_names():
    return ['Office31', 'OfficeHome', 'VisDA2017']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name in datasets.__dict__:
                                                   
        dataset = datasets.__dict__[dataset_name]

        def officehome_domains_ready(r, tasks):
            if dataset_name != 'OfficeHome':
                return False
            normalize = dataset._normalize_task
            mapping = dataset.DOMAIN_DIRS
            required_dirs = {
                mapping.get(normalize(task), None)
                for task in tasks
            }
            required_dirs.discard(None)
            if not required_dirs:
                return False
            return all(os.path.isdir(os.path.join(r, domain)) for domain in required_dirs)

        required_tasks = list(source) + list(target)
        download_flag = not (dataset_name == 'OfficeHome' and officehome_domains_ready(root, required_tasks))

        def concat_dataset(tasks, **kwargs):
            return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])

        train_source_dataset = concat_dataset(root=root, tasks=source, download=download_flag, transform=train_source_transform)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=download_flag, transform=train_target_transform)
        val_dataset = concat_dataset(root=root, tasks=target, download=download_flag, transform=val_transform)
                                         
                                                                                                                          
               
        test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(
            f"Unknown dataset '{dataset_name}'. Use one of: {get_dataset_names()}. "
            "Pass dataset root as the positional 'root' argument and dataset name via -d/--data."
        )

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names



                                                    
def get_ratio_dataset(dataset_name, root, source, target, train_source_transform, val_transform, ratio,
                      train_target_transform=None, ):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name in datasets.__dict__:
                                                   
        dataset = datasets.__dict__[dataset_name]
        dataset_ratio = datasets.__dict__[dataset_name + 'ratio']

        assert len(source) == 2, "Must have two source."

        def officehome_domains_ready(r, tasks):
            if dataset_name != 'OfficeHome':
                return False
            normalize = dataset._normalize_task
            mapping = dataset.DOMAIN_DIRS
            required_dirs = {
                mapping.get(normalize(task), None)
                for task in tasks
            }
            required_dirs.discard(None)
            if not required_dirs:
                return False
            return all(os.path.isdir(os.path.join(r, domain)) for domain in required_dirs)

        required_tasks = list(source) + list(target)
        download_flag = not (dataset_name == 'OfficeHome' and officehome_domains_ready(root, required_tasks))

        def concat_dataset(tasks, **kwargs):
            return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])

                                                         
        first_source_dataset = dataset(root=root, task=source[0], download=download_flag, transform=train_source_transform)
        baseDatasetLength = len(first_source_dataset)
        second_source_dataset = dataset_ratio(root=root, task=source[1], download=download_flag,
                                              transform=train_source_transform, baseDatasetLength=baseDatasetLength,
                                              ratio=ratio)
        train_source_dataset = ConcatDataset([first_source_dataset, second_source_dataset])
        train_target_dataset = concat_dataset(root=root, tasks=target, download=download_flag, transform=train_target_transform)
        val_dataset = concat_dataset(root=root, tasks=target, download=download_flag, transform=val_transform)
        test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError(
            f"Unknown dataset '{dataset_name}'. Use one of: {get_dataset_names()}."
        )

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

                             
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

                            
            output = model(images)
            loss = F.cross_entropy(output, target)

                                              
            acc1, = accuracy(output, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

                                  
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        if confmat:
            print(confmat.format(args.class_names))

    return top1.avg



def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


