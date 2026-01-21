import cv2
import torch
import torch.optim as optim


from config import *
from visual_helpers import visualize_result, visualize_result_w_bbs


class Optimizer:
    def __init__(self, interp):
        self.interp = interp
        self.params = {
            "INIT_RANDOM": INIT_RANDOM,
            "USE_SEED": USE_SEED,
            "LAMBDA_BASE": 1000,
            "LAMBDA_DISTANCE": 0,
            "LAMBDA_DISTANCE_OUTER": 100,
            "LAMBDA_DISTANCE_INNER": 1,
            "LAMBDA_TV": 200,
            "REGION": [0, 0, 127, 127],
            "CLASS_ID": 0
        }

        self.transforms = list(INITIAL_TRANSFORMS)

        self.initial = None

    def set_initial(self):
        if self.params["INIT_RANDOM"]:
            self.initial = (torch.randn((1, 3, self.interp.img_shape[0], self.interp.img_shape[1]))
                                  .to(self.interp.device))
        else:
            self.initial = self.interp.seed

        if not self.params["USE_SEED"]:
            self.params["LAMBDA_DISTANCE_OUTER"] = 0
            self.params["LAMBDA_DISTANCE_INNER"] = 0

    def run(self, num_iterations, lr, show=False, printing=False, progress_callback=None):

        self.interp.curr_x = self.initial.detach().clone().to(self.interp.device).requires_grad_(True)

        optimizer = optim.Adam([self.interp.curr_x], lr=lr)

        if self.interp.targets.get is not None:
            with torch.no_grad():
                self.interp.model.model(self.interp.curr_x)
                orig_activations = self.interp.targets.get()

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            x_transformed = self.interp.curr_x.to(self.interp.device)
            for transform in self.transforms:
                x_transformed = transform.apply(x_transformed, self.params)

            out = self.interp.model.model(x_transformed)
            # print(type(out), len(out), out[0], type(out[0]), out[0].shape)

            loss = self.interp.loss.get(x_transformed, out, self.params)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x.clamp_(0, 1)

            # prints
            if printing and iteration % 20 == 0:
                print(f"Iteration {iteration}: Loss = {loss}")

            if progress_callback is not None:
                progress_callback(iteration, num_iterations)

        if self.interp.targets.get is not None:
            with torch.no_grad():
                self.interp.model.model(self.interp.curr_x)
                post_activations = self.interp.targets.get()

            if printing:
                change = torch.sum(post_activations - orig_activations)
                print("Change in activations:", change)

        if show:
            visualize_result_w_bbs(self.interp.model, self.interp.curr_x)

        return self.interp.curr_x
