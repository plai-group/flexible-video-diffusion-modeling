from email.mime import base
import os
from pathlib import Path
import argparse
import numpy as np
import torch as th
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import wandb

from improved_diffusion.script_util import str2bool

import multiprocessing as mp
from functools import partial


def video_to_image(fname, video_path, frame_path):
    coords_fname = fname.replace('.pt', '.npy').replace('video_', 'coords_')
    video = th.load(video_path / fname).numpy()
    coords = np.load(video_path / coords_fname)
    print("Processing video:", str(video_path / fname))
    for frame_idx, (frame, coord) in enumerate(zip(video, coords)):
        frame_fname = fname.replace('.pt', f'_frame_{frame_idx}.npy')
        coord_fname = coords_fname.replace('.npy', f'_frame_{frame_idx}.npy')
        np.save(frame_path / frame_fname, frame)
        np.save(frame_path / coord_fname, coord)


def get_cell(coord):
    count, _, _ = np.histogram2d([coord[0]], [coord[1]], bins=10, range=[[-10,400], [-10,400]])  # range is specific to Carla Town01
    cell = count.flatten().nonzero()[0]
    return cell


class CarlaRegressorDataset(th.utils.data.Dataset):

    def __init__(self, train, path, transforms=None):
        """
        Uses video directory like
        ```
        dataset/
            video_0.pt
            coords_0.npy
            ...
        ```
        Videos in this directory are split and saved into a subfolder like:
        ```
        dataset/
            individual-frames/
                train/
                    video_0_frame_0.npy
                    coords_0_frame_0.npy
                    video_0_frame_1.npy
                    ...
                test/
                    ...
        ```
        """
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.video_path = Path(path)
        self.path  = self.video_path / 'individual-frames' / ("train" if train else "test")
        video_split_path = self.video_path / f"video_{'train' if train else 'test'}.csv"
        self.video_fnames = [line.rstrip('\n').split('/')[-1] for line in open(video_split_path, 'r').readlines() if '.pt' in line]
        self.videos_to_images()
        self.paths = list(self.path.glob("video_*.npy"))

    def videos_to_images(self):
        self.path.mkdir(exist_ok=True, parents=True)
        # count number of files matching globstring
        videos_done = len(list(self.path.glob("video_*_frame_0.npy")))
        if videos_done == len(self.video_fnames):
            return
        #for fname in self.video_fnames:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except:
            n_cpus = mp.cpu_count()
        mp.Pool(processes=n_cpus).map(
            partial(video_to_image, video_path=self.video_path, frame_path=self.path),
            self.video_fnames
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.paths[idx]
        target_path = str(img_path).replace('video', 'coords')
        img = np.load(img_path)
        target = np.load(target_path)
        target = target[[0, 1]]  # use only 2D position
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target, get_cell(target)





with_classifier = False
model = 'resnet18'
with_transforms = True


base_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_dataloaders(data_dir, with_transforms, batch_size):

    if with_transforms:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ColorJitter(brightness=.1, hue=.1),
            base_data_transform,
        ])
    else:
        train_transform = base_data_transform

    train_dataset = CarlaRegressorDataset(train=True, path=data_dir, transforms=train_transform)
    test_dataset = CarlaRegressorDataset(train=False, path=data_dir, transforms=base_data_transform)

    make_loader = lambda dataset: th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_loader = make_loader(train_dataset)
    test_loader = make_loader(test_dataset)

    return {'train': train_loader, 'test': test_loader}


class MultiHeadEfficientNet_b7(nn.Module):

    @staticmethod
    def make_last_layer(in_dim, out_dim):
        return nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_dim, out_dim))

    def __init__(self, pretrained=True):
        super().__init__()
        self.efficientnet_b7 = torchvision.models.efficientnet_b7(pretrained=pretrained)
        self.efficientnet_b7.classifier = nn.Identity()
        self.regressors = nn.ModuleList([self.make_last_layer(2560, 2) for i in range(100)])

    def forward(self, inputs, cells):
        emb = self.efficientnet_b7(inputs)
        coords = []
        for idx, cell in enumerate(cells):
            coords.append(self.regressors[cell](emb[idx]))
        coords = th.stack(coords)
        return coords


class MultiHeadResNet152(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet152(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.regressors = nn.ModuleList([nn.Linear(in_features, 2) for i in range(100)])

    def forward(self, inputs, cells):
        emb = self.resnet(inputs)
        coords = []
        for idx, cell in enumerate(cells):
            coords.append(self.regressors[cell](emb[idx]))
        coords = th.stack(coords)
        return coords


def get_resnet152_classifier(pretrained=True):
    model = torchvision.models.resnet152(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)
    return model


def get_efficientnet_b7_classifier(pretrained=True):
    model = torchvision.models.efficientnet_b7(pretrained=pretrained)
    model.classifier = nn.Linear(2560, 100)
    return model


def set_up_model(is_classifier, model_name, device, pretrained=True):
    if is_classifier and model_name == 'resnet152':
        model_conv = get_resnet152_classifier(pretrained=pretrained)
    elif is_classifier and model_name == 'efficientnet_b7':
        model_conv = get_efficientnet_b7_classifier(pretrained=pretrained)
    elif model_name == 'resnet152':
        model_conv = MultiHeadResNet152(pretrained=pretrained)
    elif model_name == 'efficientnet_b7':
        model_conv = MultiHeadEfficientNet_b7(pretrained=pretrained)
    else:
        raise ValueError('Unknown model')
    return model_conv.to(device)


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet152', help='Model to use')
    parser.add_argument('--is_classifier', type=str2bool, default=False, help='Train classifier vs the multi-headed regressor.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--with_transforms', type=str2bool, default=True)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    return parser.parse_args()


def train():
    args = parse_train_args()
    wandb.init(project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'],
               config=args)

    args.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    model       = set_up_model(args.is_classifier, args.model, args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Decay LR by a factor of 0.1 every 7 epochs
    dataloaders = make_dataloaders(args.data_dir, args.with_transforms, args.batch_size)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.float('inf')

    for epoch in range(args.num_epochs):
        losses = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, coords, cells in dataloaders[phase]:
                inputs = inputs.to(args.device)
                coords = coords.to(args.device).float()
                cells = cells.to(args.device)

                optimizer.zero_grad()

                with th.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) if args.is_classifier else model(inputs, cells)
                    if args.is_classifier:
                        loss = nn.BCELoss()(nn.Sigmoid()(outputs), cells)
                    else:
                        loss  = nn.MSELoss()(outputs, coords)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            losses[phase] = epoch_loss

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                th.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pth"))
                wandb.save(os.path.join(wandb.run.dir, f"model_{epoch}.pth"))


        wandb.log({f"{k}_loss": v for k, v in losses.items()})

    print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def load_classifier_regressor_like_paper(classifier_path, regressor_path, device):
    classifier = set_up_model(is_classifier=True, model_name='resnet152', device=device, pretrained=False)
    regressor = set_up_model(is_classifier=False, model_name='resnet152', device=device, pretrained=False)
    classifier.load_state_dict(th.load(classifier_path))
    regressor.load_state_dict(th.load(regressor_path))
    classifier.eval()
    regressor.eval()
    return classifier.to(device), regressor.to(device)


@th.no_grad()
def predict_coord_batch(frames, classifier, regressor):
    orig_device = frames.device
    device = next(classifier.parameters()).device
    frames = frames.to(device)
    cells = classifier(frames).argmax(dim=1)
    return regressor(frames, cells).to(orig_device)


def predict_coords(frames, classifier, regressor, batch_size):
    coords = []
    while len(frames) > 0:
        some_frames = frames[:batch_size]
        frames = frames[batch_size:]
        coords.append(predict_coord_batch(some_frames, classifier, regressor))
    return th.cat(coords, dim=0)