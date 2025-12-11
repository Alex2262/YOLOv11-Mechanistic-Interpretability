
from ultralytics import YOLO
import torch

from PIL import Image
from torchvision import transforms as T


from loss_computer import LossComputer
from optimizer import Optimizer
from target_manager import TargetManager
from visual_helpers import visualize_layer_filters


class YoloInterp:

    def __init__(self, device):
        self.device = device
        self.model = YOLO("yolo11n.pt").to(device)
        self.model.model.eval()

        for param in self.model.model.parameters():
            param.requires_grad = False

        self.layers = self.model.model.model

        self.conv0 = self.layers[0]
        self.conv1 = self.layers[1]

        self.activations = {}
        self.seed = None
        self.curr_x = None

        self.img_shape = (128, 128)

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

        print(self.model.model.stride)
        # visualize_layer_filters(self.layers[5].conv.weight.data.clone())

    def set_seed(self, path):
        image = Image.open(path)
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        t = transform(image).unsqueeze(dim=0).to(self.device)

        self.seed = t.clone().detach().requires_grad_(True).to(self.device)

    def compare_activations(self, images):
        for image_path in images:
            image = Image.open(image_path)
            transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
            x = transform(image).unsqueeze(dim=0).to(self.device)

            self.model.model(x)
            acts = self.targets.get()

            print("Image:", image_path, acts.sum())
