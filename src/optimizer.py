
import torch
import torch.optim as optim
from torchvision import transforms as T
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, interp):
        self.interp = interp

    def run(self, num_iterations, lr=0.01):

        if self.interp.seed is None:
            raise RuntimeError("No seed image set")

        x = self.interp.seed
        optimizer = optim.Adam([x], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            out = self.interp.model.model(x)
            loss = self.interp.loss.get()

            loss.backward()
            optimizer.step()

            # clamp img to valid range
            with torch.no_grad():
                x.clamp_(0, 1)

            # prints
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

        self.visualize_result(x)

    @staticmethod
    def visualize_result(x):
        optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        plt.imshow(optim_img)
        plt.axis('off')
        plt.title("Optimized Image")
        plt.show()

