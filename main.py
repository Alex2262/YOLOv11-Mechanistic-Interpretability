from ultralytics import YOLO
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


class YoloInterp:

    def __init__(self, device):
        self.device = device
        self.model = YOLO("yolo11n.pt").to(device)
        self.model.model.eval()
        self.model.model.to(device)

        self.layers = self.model.model.model

        self.conv0 = self.layers[0]
        self.conv1 = self.layers[0]

        self.get_targets = None  # a function that returns the activations for the neurons we're interested in
        self.seed = None

        self.activations = {}

    # ---- HELPER METHODS ----
    # random helper methods

    def register_hook(self, module, hook_name):
        def hook(mod, inp, out):
            self.activations[hook_name] = out

        module.register_forward_hook(hook)

    def print_layers(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer.__class__.__name__}")

    def set_seed(self, path):
        image = Image.open(path)
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        t = transform(image).unsqueeze(dim=0).to(self.device)

        self.seed = t.clone().detach().requires_grad_(True)

    # ---- CONFIGURATION METHODS ----
    # Only these methods need to know about the YOLO layer structure, so the rest of the project
    # doesn't need layer specifics and everything else can be abstracted
    #

    def set_targets_conv_layer(self, layer_idx, channels: None | int | list[int]):
        """
        For Conv layers:
        self.activations["conv#"] has dimensions (B, C, H, W)
        batch should be 0 since there's only 1 image
        C is the channel

        Note that in Yolo11 these are the conv layers:
        0, 1, 3, 5, 7, 17, 20
        """

        module = self.layers[layer_idx]
        hook_name = f"conv{layer_idx}"

        self.register_hook(module, hook_name)

        def get_targets_conv_helper():
            act = self.activations[hook_name]
            if channels is None:  # Case when we want all channels
                return act[0]
            elif isinstance(channels, int):  # Case when we want only 1 specific channel
                return act[0, channels]
            elif isinstance(channels, list):  # Case when we want multiple channels
                return act[0, channels]

        self.get_targets = get_targets_conv_helper

    def set_targets_conv0(self, channels: None | int | list[int]):
        self.set_targets_conv_layer(0, channels)

    # ---- LOSS FUNCTIONS ----
    #

    def get_loss_basic(self):
        return -self.get_targets().mean()

    def get_loss_seed(self):
        # TODO: IMPL THIS
        return -self.get_targets().mean()

    # ---- ACTUAL ALGO ----
    # just running Adam on this

    def run(self, num_iterations):
        x = self.seed
        optimizer = optim.Adam([x], lr=0.1)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            out = self.model.model(x)
            loss = self.get_loss_basic()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x.clamp_(0, 1)

            if iteration % 10 == 0:
                print(f"Loss at iteration {iteration}: {loss}")

        optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        plt.imshow(optim_img)
        plt.axis('off')
        plt.show()


def main():
    y = YoloInterp(device="cpu")

    y.print_layers()

    y.set_seed("dog.jpg")
    y.set_targets_conv0(13)
    y.run(20)


if __name__ == '__main__':
    main()
