from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data import Dataset

from datasets import FrameVideoDataset

class AugmentedVideoDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.augmentation = T.Compose([
            T.RandomResizedCrop(
                size=(64, 64),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05)
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, label = self.dataset[idx]

        # [C, T, H, W] -> [T, C, H, W]
        frames = video.permute(1, 0, 2, 3)

        # Apply the same random transformations
        # to every frame in this video.
        frames = self.augmentation(frames)

        # [T, C, H, W] -> [C, T, H, W]
        video = frames.permute(1, 0, 2, 3).contiguous()

        return video, label


def get_video_loaders(
    root_dir='/dtu/datasets1/02516/ufc10',
    batch_size=16,
    num_workers=4
):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    # Training dataset
    train_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        stack_frames=True
    )
    """
    # Apply video-level augmentation
    train_dataset = AugmentedVideoDataset(
        train_dataset
    )
    """
    # Validation dataset (no augmentation)
    val_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        stack_frames=True
    )

    # Test dataset (no augmentation)
    test_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='test',
        transform=transform,
        stack_frames=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader