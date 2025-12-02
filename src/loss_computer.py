

class LossComputer:
    def __init__(self, interp):
        self.interp = interp
        self.get = self.get_basic

    def get_basic(self):
        if self.interp.targets.get is None:
            raise RuntimeError("No targets configured")

        targets = self.interp.targets.get()
        activation_loss = -targets.mean()

        return activation_loss

    def get_seeded(self):
        # TODO: impl this
        pass
