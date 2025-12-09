
import math
import torch.nn.functional as F
from config import *
from region_maxxing import get_region_loss, REGION

LAMBDA_TV = 30  # 1


def tv_loss(img):
    # ok so (1, 3, H, W)
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return torch.abs(dx).mean(dim=(1, 2, 3)) + torch.abs(dy).mean(dim=(1, 2, 3))


def regularization(img):
    loss = LAMBDA_TV * tv_loss(img)
    return loss


class LossComputer:
    def __init__(self, interp):
        self.interp = interp
        # self.get = self.get_seeded if USE_SEED else self.get_basic
        self.get = self.get_region_and_layer
        # self.get = self.get_region

    def get_basic(self, inp, out, params):
        if self.interp.targets.get is None:
            raise RuntimeError("No targets configured")

        targets = self.interp.targets.get()
        if isinstance(targets, list):
            activation_loss = 0
            s = 0
            for target in targets:
                s += target.numel()

            for target in targets:
                activation_loss += -target.sum()

            activation_loss /= s

        else:
            activation_loss = -targets.mean()

        reg_loss = regularization(inp)[0]

        return activation_loss + reg_loss

    def get_seeded(self, inp, out, params):
        base_loss = self.get_basic(inp, out, params)

        distance = torch.norm(inp - self.interp.seed)

        loss = base_loss + 0.02 * distance

        # print(distance, activation_loss, loss)

        return loss

    def get_region(self, inp, out, params):

        region_loss = get_region_loss(inp, out)

        mask = torch.ones_like(inp)
        x1, y1, x2, y2 = REGION
        mask[:, :, y1:y2 + 1, x1:x2 + 1] = 0

        regularization_loss = regularization(inp)[0]

        # distance = torch.norm((inp - self.interp.seed) * mask)

        if USE_SEED:
            distance = torch.norm((inp - self.interp.seed) * mask)
        else:
            distance = 0

        return region_loss + regularization_loss + 10 * distance

    def get_region_and_layer(self, inp, out, params):
        region_loss = get_region_loss(inp, out)
        base_loss = self.get_basic(inp, out, params)

        mask = torch.ones_like(inp)
        x1, y1, x2, y2 = REGION
        mask[:, :, y1:y2 + 1, x1:x2 + 1] = 0
        distance = torch.norm((inp - self.interp.seed) * mask)

        return region_loss + params["LAMBDA_BASE"] * base_loss + params["LAMBDA_DISTANCE"] * distance
