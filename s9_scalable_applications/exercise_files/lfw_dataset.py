"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.path_to_folder = path_to_folder
        self.transform = transform

    def __len__(self):
        # You can implement this if needed
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        # Load the image from file using PIL or any other method you prefer
        img = load_image_from_file(self.path_to_folder, index)
        return self.transform(img)

def load_image_from_file(path_to_folder, index):
    # Implement a function to load the image from file (e.g., using PIL)
    # Replace this with your own logic to load images
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="s9_scalable_applications/lfw", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=4, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        dataiter = iter(dataloader)
        images, labels = next(dataiter)

        def show_pil_images(images):
            for image in images:
                image = transforms.ToPILImage()(image)
                image.show()

        show_pil_images(images)

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")
