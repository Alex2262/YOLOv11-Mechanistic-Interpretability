
import torch
from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def apply(self, x, params):
        pass


class Jitter(Transform):
    def __init__(self, amount: int):
        self.amount = amount
        self.shift_y = None
        self.shift_x = None

    def apply(self, x, params):
        self.shift_y = torch.randint(-self.amount, self.amount + 1, (1,)).item()
        self.shift_x = torch.randint(-self.amount, self.amount + 1, (1,)).item()

        x_transformed = torch.roll(x, shifts=(self.shift_y, self.shift_x), dims=(2, 3))
        return x_transformed


class RegionJitter(Transform):
    def __init__(self, amount: int):
        self.amount = amount
        self.shift_y = None
        self.shift_x = None

    def apply(self, x, params):
        self.shift_y = torch.randint(-self.amount, self.amount + 1, (1,)).item()
        self.shift_x = torch.randint(-self.amount, self.amount + 1, (1,)).item()

        x1, y1, x2, y2 = params["REGION"]

        out = x.clone()

        region = x[:, :, y1:y2, x1:x2]

        jittered_region = torch.roll(region, shifts=(self.shift_y, self.shift_x), dims=(2, 3))

        out[:, :, y1:y2, x1:x2] = jittered_region
        return out
