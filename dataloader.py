from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameVideoDataset


def get_video_loaders(
    root_dir='/dtu/datasets1/02516',
    batch_size=16,
    num_workers=4
):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    train_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        stack_frames=True
    )

    val_dataset = FrameVideoDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        stack_frames=True
    )

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

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_video_loaders()

    videos, labels = next(iter(train_loader))

    print("Video batch shape:", videos.shape)
    print("Label batch shape:", labels.shape)