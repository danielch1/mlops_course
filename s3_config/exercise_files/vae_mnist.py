"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image


@hydra.main(config_name="config.yaml")
def train(config):
    print(f"Configuration: {config}")

    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(
        config.hparams.dataset_path,
        transform=mnist_transform,
        train=True,
        download=True,
    )
    test_dataset = MNIST(
        config.hparams.dataset_path,
        transform=mnist_transform,
        train=False,
        download=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config.hparams.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config.hparams.batch_size, shuffle=False
    )

    encoder = Encoder(
        input_dim=config.hparams.x_dim,
        hidden_dim=config.hparams.hidden_dim,
        latent_dim=config.hparams.latent_dim,
    )
    decoder = Decoder(
        latent_dim=config.hparams.latent_dim,
        hidden_dim=config.hparams.hidden_dim,
        output_dim=config.hparams.x_dim,
    )

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=config.hparams.lr)

    print("Start training VAE...")
    model.train()
    for epoch in range(config.hparams.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(config.hparams.batch_size, config.hparams.x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*config.hparams.batch_size)}"
        )
    print("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(config.hparams.batch_size, config.hparams.x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(config.hparams.batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(config.hparams.batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(config.hparams.batch_size, config.hparams.latent_dim).to(
            DEVICE
        )
        generated_images = decoder(noise)

    save_image(
        generated_images.view(config.hparams.batch_size, 1, 28, 28),
        "generated_sample.png",
    )


if __name__ == "__main__":
    train()
