
from ultralytics import YOLO
import torch

from PIL import Image
from torchvision import transforms as T


from src.loss_computer import LossComputer
from src.optimizer import Optimizer
from src.target_manager import TargetManager


class YoloInterp:

    def __init__(self, device):
        self.device = device
        self.model = YOLO("yolo11n.pt").to(device)
        self.model.model.eval()
        self.model.model.to(device)

        self.layers = self.model.model.model

        self.conv0 = self.layers[0]
        self.conv1 = self.layers[1]

        self.activations = {}
        self.seed = None
        self.curr_x = None

        # Abstracted Managers
        self.targets = TargetManager(self)
        self.loss = LossComputer(self)
        self.optimizer = Optimizer(self)

    #
    # ---- HELPER METHODS ----
    # random helper methods
    #

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