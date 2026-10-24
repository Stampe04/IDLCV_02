from datasets import FrameVideoDataset, FrameImageDataset

root_dir = "/dtu/datasets1/02516/ucf101_noleakage"
transform = None  # Define your transformations here if needed

# Training dataset
train_dataset = FrameImageDataset(
    root_dir=root_dir,
    split='train',
    transform=transform,
    flows = False
)

print(len(train_dataset))