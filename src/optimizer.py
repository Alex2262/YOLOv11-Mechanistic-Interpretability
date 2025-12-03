import torch
import torch.optim as optim
from torchvision import transforms as T
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, interp):
        self.interp = interp
    
    @staticmethod
    def jitter(x, amount=4):
        shift_y = torch.randint(-amount, amount + 1, (1,)).item()
        shift_x = torch.randint(-amount, amount + 1, (1,)).item()

        x_j = torch.roll(x, shifts=shift_y, dims=2)
        x_j = torch.roll(x_j, shifts=shift_x, dims=3)
        return x_j, shift_y, shift_x

    @staticmethod
    def unjitter(x, shift_y, shift_x):
        x_uj = torch.roll(x, shifts=-shift_y, dims=2)
        x_uj = torch.roll(x_uj, shifts=-shift_x, dims=3)
        return x_uj

    def run(self, num_iterations, lr=0.01):

        if self.interp.seed is None:
            raise RuntimeError("No seed image set")

        
        self.interp.curr_x = (self.interp.seed.detach().clone().to(self.interp.device).requires_grad_(True))

        optimizer = optim.Adam([self.interp.curr_x], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            x_jit, sy, sx = self.jitter(self.interp.curr_x)

            # use jittered image
            out = self.interp.model.model(x_jit)
            loss = self.interp.loss.get()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x[:] = self.unjitter(self.interp.curr_x, sy, sx)
                self.interp.curr_x.clamp_(0, 1)

            # prints
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

        self.visualize_result(self.interp.curr_x)

    @staticmethod
    def visualize_result(x):
        optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
        plt.imshow(optim_img)
        plt.axis('off')
        plt.title("Optimized Image")
        plt.show()
