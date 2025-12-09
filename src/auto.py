

import torch

from loss_computer import LossComputer, regularization
from config import INITIAL_TRANSFORMS
from transform import Jitter
from yolo_interp import YoloInterp

from region_maxxing import verify, get_region_loss, REGION

import torch.optim as optim


class Auto:

    def __init__(self, k):
        self.interp = YoloInterp(device="cpu")
        self.k = k

        self.params = {
            "LAMBDA_BASE": 10,
            "LAMBDA_DISTANCE": 1
        }

        self.transforms = list(INITIAL_TRANSFORMS)

    def run_phase1(self, layers):
        layer_channel_indices = []
        for layer in layers:
            layer_channel_indices.append((layer, None))

        self.interp.targets.set_layers(layer_channel_indices)

        with torch.no_grad():
            self.interp.model.model(self.interp.seed)
            orig_activations = self.interp.targets.get()

        self.interp.optimizer.run(1000, 0.005)

        with torch.no_grad():
            self.interp.model.model(self.interp.curr_x)
            post_activations = self.interp.targets.get()

        new_layers = {}

        for i in range(len(post_activations)):
            layer_idx = layers[i]
            layer_change = post_activations[i] - orig_activations[i]
            change_channels = torch.sum(layer_change, dim=(1, 2))

            change_list = []

            # s = 0
            # for channel in range(change_channels.shape[0]):
            #     s += abs(change_channels[channel].item())

            for channel in range(change_channels.shape[0]):
                score = change_channels[channel].item()  # / s
                change_list.append((score, layer_idx, channel))

            change_list = sorted(change_list, key=lambda x: x[0], reverse=True)

            for j in range(min(self.k, change_channels.shape[0])):
                change = change_list[j][0]
                channel = change_list[j][2]

                print(change, layer_idx, channel)
                if layer_idx not in new_layers:
                    new_layers[layer_idx] = []

                new_layers[layer_idx].append(channel)

        new_layer_channel_indices = []
        for layer, channel_indices in new_layers.items():
            new_layer_channel_indices.append((layer, channel_indices))

        return new_layer_channel_indices

    def run_phase2(self, layer_channel_indices):
        self.interp.loss.get = self.interp.loss.get_basic
        self.interp.targets.set_layers(layer_channel_indices)
        self.interp.optimizer.run(1000, 0.005)

    def auto_layer(self, layers):
        self.interp.set_seed("images/road.jpg")
        layer_channel_indices = self.run_phase1(layers)

        print("Found", layer_channel_indices)

        self.run_phase2(layer_channel_indices)

    def bin(self, layer, channel):
        def test():
            self.interp.set_seed("images/road.jpg")
            self.interp.targets.set_conv_layer(layer, channel)

            with torch.no_grad():
                self.interp.model.model(self.interp.seed)
                orig_activations = self.interp.targets.get()

            x = self.interp.optimizer.run(200, 0.01)

            with torch.no_grad():
                out = self.interp.model.model(x)
                post_activations = self.interp.targets.get()

            c = torch.sum(post_activations - orig_activations).item()

            p = verify(x, out).item()

            if p >= 0.1:
                return True, p, c
            else:
                return False, p, c

        lo = 0
        hi = 1000
        eps = 1

        self.interp.optimizer.params["LAMBDA_BASE"] = hi
        orig_change = test()[2]

        best_scale = 0
        best_change = 0

        while lo + eps < hi:
            mid = (lo + hi) / 2

            print("RANGE:", lo, mid, hi)

            self.interp.optimizer.params["LAMBDA_BASE"] = mid

            flag, prob, change = test()

            print(flag, prob, change)

            if flag:
                lo = mid
                best_scale = mid
                best_change = change
            else:
                hi = mid

        prop_change = best_change / orig_change

        print(best_scale, best_change, orig_change, prop_change)
        return prop_change

    def channel_loss(self, c, inp, out, lambda_bases):
        targets = self.interp.targets.get()[c]

        activation_loss = -targets.mean()
        region_loss = get_region_loss(inp, out, c)

        return lambda_bases[c] * activation_loss + region_loss

    def batched_loss(self, b, inp, out, lambda_bases):
        losses = []

        for c in range(b):
            losses.append(self.channel_loss(c, inp, out, lambda_bases))

        batched_loss = torch.stack(losses)

        regularization_loss = lambda_bases * regularization(inp)

        mask = torch.ones_like(inp)
        x1, y1, x2, y2 = REGION
        mask[:, :, y1:y2 + 1, x1:x2 + 1] = 0
        distance = torch.linalg.vector_norm((inp - self.interp.seed) * mask, dim=(1, 2, 3))

        distance_loss = self.params["LAMBDA_DISTANCE"] * distance

        batched_loss += regularization_loss + distance_loss

        return batched_loss

    def run_batched(self, b, num_iterations, lr, lambda_bases):
        self.interp.curr_x = (self.interp.seed.repeat(b, 1, 1, 1)
                              .detach().clone().to(self.interp.device).requires_grad_(True))

        optimizer = optim.Adam([self.interp.curr_x], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            x_transformed = self.interp.curr_x.to(self.interp.device)
            for transform in self.transforms:
                x_transformed = transform.apply(x_transformed)

            out = self.interp.model.model(x_transformed)
            loss = self.batched_loss(b, x_transformed, out, lambda_bases)

            print(loss)

            loss.backward(gradient=torch.ones_like(loss))

            optimizer.step()

            with torch.no_grad():
                self.interp.curr_x.clamp_(0, 1)

        return self.interp.curr_x

    def bin_layer(self, layer):
        num_channels = self.interp.layers[layer].conv.weight.shape[0]

        def test(lambda_bases):
            self.interp.set_seed("images/road.jpg")
            self.interp.targets.set_conv_layer(layer, None)

            with torch.no_grad():
                self.interp.model.model(self.interp.seed)
                orig_activations = self.interp.targets.get()

            x = self.run_batched(num_channels, 200, 0.01, lambda_bases)

            with torch.no_grad():
                out = self.interp.model.model(x)
                post_activations = self.interp.targets.get()

            chg = torch.sum(post_activations - orig_activations, dim=(1,2,3))

            ps = []
            for channel in range(num_channels):
                ps.append(verify(x, out, channel).item())

            return ps, chg

        eps = 1
        los = [0] * num_channels
        his = [1000] * num_channels

        orig = test(torch.tensor(his, device=self.interp.device))[1]
        best = torch.zeros(num_channels)

        mids = [0] * num_channels

        while True:
            all_done = True

            for c in range(num_channels):
                if los[c] + eps < his[c]:
                    all_done = False
                    mids[c] = (los[c] + his[c]) / 2

            if all_done:
                break

            print(los)
            print(his)

            lb = torch.tensor(mids, device=self.interp.device)
            p_list, chg_tensor = test(lb)

            for c in range(num_channels):
                if p_list[c] >= 0.1:
                    los[c] = mids[c]
                    best[c] = chg_tensor[c]
                else:
                    his[c] = mids[c]

        prop = best / orig
        print(best, orig, prop)

    def search_layer(self, layer):
        num_channels = self.interp.layers[layer].conv.weight.shape[0]

        probs = []

        for channel in num_channels:
            p = self.bin(layer, channel)
            probs.append([p, channel])

        sorted(probs, key=lambda x: x[0], reverse=True)






