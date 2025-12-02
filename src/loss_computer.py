import torch


class LossComputer:
    def __init__(self, interp):
        self.interp = interp
        self.get = self.get_seeded

    def get_basic(self):
        if self.interp.targets.get is None:
            raise RuntimeError("No targets configured")

        targets = self.interp.targets.get()
        activation_loss = -targets.mean()

        return activation_loss

    def get_seeded(self):
        activation_loss = self.get_basic()

        distance = torch.norm(self.interp.curr_x - self.interp.seed)

        loss = activation_loss + 0.02 * distance

        # print(distance, activation_loss, loss)

        return loss
