from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets import FrameVideoDataset, FrameImageDataset, TwoStreamDataset


def get_video_loaders(
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
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


def get_flows_loaders(
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
    batch_size=16,
    num_workers=4
):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    train_dataset = FrameImageDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        flows=True
    )

    val_dataset = FrameImageDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        flows=True
    )

    test_dataset = FrameImageDataset(
        root_dir=root_dir,
        split='test',
        transform=transform,
        flows=True
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

def get_twostream_loaders(root_dir='/dtu/datasets1/02516/ucf101_noleakage',
                          batch_size=16, num_workers=4,
                          flow_channels=3, n_flows=9):
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    kw = dict(root_dir=root_dir, transform=transform,
              flow_channels=flow_channels, n_flows=n_flows)

    train_ds = TwoStreamDataset(split='train', **kw)
    val_ds   = TwoStreamDataset(split='val', **kw)
    test_ds  = TwoStreamDataset(split='test', **kw)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )

if __name__ == "__main__":
    train_video_loader, val_video_loader, test_video_loader = get_video_loaders()
    train_flow_loader, val_flow_loader, test_flow_loader = get_flows_loaders()

    videos, labels_videos = next(iter(train_video_loader))
    flows, labels_flows = next(iter(train_flow_loader))

    print("Video batch shape:", videos.shape)
    print("Flow batch shape:", flows.shape)
    print("Label batch shape:", labels_videos.shape)
    print("Label batch shape:", labels_flows.shape)