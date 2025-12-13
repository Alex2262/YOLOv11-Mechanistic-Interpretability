import matplotlib
import threading

import torch

from yolo_interp import YoloInterp
from ultralytics import YOLO

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

import tkinter as tk
from tkinter import filedialog
from PIL import Image
from torchvision import transforms as T


matplotlib.use("TkAgg")


def get_np(x_tensor):
    return (
        x_tensor.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )


###############################################################################
# REGION SELECTOR (mouse drag)
###############################################################################

class RegionSelector:
    def __init__(self, ax):
        self.ax = ax
        self.disabled = False

        self.temp_rect = None       # rectangle while dragging
        self.final_rect = None      # frozen rectangle after release
        self.x0 = None
        self.y0 = None
        self.selected_region = None

        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_move = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)

    def clear_rectangles(self):
        """Remove any existing rectangles so only one appears."""
        if self.temp_rect is not None:
            try:
                self.temp_rect.remove()
            except ValueError:
                pass
        if self.final_rect is not None:
            try:
                self.final_rect.remove()
            except ValueError:
                pass
        self.temp_rect = None
        self.final_rect = None

    def on_press(self, event):
        if self.disabled:
            return

        if event.inaxes != self.ax:
            return

        self.x0, self.y0 = event.xdata, event.ydata

        # Remove any old rectangles
        self.clear_rectangles()

        # Create temporary drag-rectangle
        self.temp_rect = plt.Rectangle(
            (self.x0, self.y0), 0, 0,
            fill=False, edgecolor="yellow", linewidth=1
        )
        self.ax.add_patch(self.temp_rect)

    def on_move(self, event):
        if self.disabled:
            return

        # Only update while dragging
        if self.temp_rect is None:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x1, y1 = event.xdata, event.ydata
        self.temp_rect.set_width(x1 - self.x0)
        self.temp_rect.set_height(y1 - self.y0)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if self.disabled:
            return

        if self.temp_rect is None:
            return

        # Determine final coords safely
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            x1 = self.x0 + self.temp_rect.get_width()
            y1 = self.y0 + self.temp_rect.get_height()
        else:
            x1, y1 = event.xdata, event.ydata

        # Normalize coordinates
        x_start = min(self.x0, x1)
        y_start = min(self.y0, y1)
        x_end   = max(self.x0, x1)
        y_end   = max(self.y0, y1)

        self.selected_region = (x_start, y_start, x_end, y_end)
        print("Selected region:", self.selected_region)

        # Replace temp rectangle with a frozen final rectangle
        try:
            self.temp_rect.remove()
        except ValueError:
            pass

        self.final_rect = plt.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            fill=False, edgecolor="yellow", linewidth=1
        )
        self.ax.add_patch(self.final_rect)

        # IMPORTANT: stop movement updates
        self.temp_rect = None

        self.ax.figure.canvas.draw_idle()

###############################################################################
# MAIN GUI APPLICATION
###############################################################################


class InterpretabilityGUI:
    def __init__(self, height, width, image_path):
        self.interp = YoloInterp(device="cpu")
        self.interp.img_shape = (height, width)
        self.interp.optimizer.params["USE_SEED"] = True
        self.interp.optimizer.params["INIT_RANDOM"] = False
        self.interp.loss.get = self.interp.loss.get_region

        self.yolo_pred = YOLO("yolo11n.pt").to(self.interp.device)

        self.height = height
        self.width = width

        self.x_tensor = None
        self.selected_region = None
        self.class_C = None

        # Hyperparameters
        self.num_iterations = 200
        self.lr = 0.01
        self.lambda_tv = 200
        self.lambda_distance_inner = 1.0
        self.lambda_distance_outer = 100.0

        self.fig = None
        self.ax = None

        self.region_selector = None

        self.show_predictions = True

        self.toggle_button = None
        self.progress_bar = None
        self.class_box = None
        self.slider_iter = None
        self.slider_lr = None
        self.slider_tv = None
        self.slider_dist_outer = None
        self.slider_dist_inner = None
        self.run_button = None
        self.load_button = None

        self.running = False

        self.build_gui()
        self.load_path(image_path)

    def build_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.3)  # space for widgets

        self.ax.set_title("YOLO Interpretability GUI")

        # Region selector
        self.region_selector = RegionSelector(self.ax)

        # Progress bar (0 to 1)
        ax_progress = plt.axes((0.4, 0.20, 0.5, 0.03))
        ax_progress.set_xticks([])
        ax_progress.set_yticks([])
        ax_progress.set_xlim(0, 1)
        ax_progress.set_ylim(0, 1)
        ax_progress.set_frame_on(True)

        self.progress_bar = plt.Rectangle(
            (0, 0), 0, 1,
            color="green"
        )
        ax_progress.add_patch(self.progress_bar)

        # --- num_iterations slider ---
        ax_iter = plt.axes((0.4, 0.16, 0.5, 0.03))
        self.slider_iter = Slider(ax_iter, "Iterations", 1, 1000, valinit=self.num_iterations)
        self.slider_iter.on_changed(self.set_iterations)

        # --- learning rate slider ---
        ax_lr = plt.axes((0.4, 0.12, 0.5, 0.03))
        self.slider_lr = Slider(ax_lr, "LR", 0.0001, 0.1, valinit=self.lr)
        self.slider_lr.on_changed(self.set_lr)

        # --- lambda_tv ---
        ax_tv = plt.axes((0.4, 0.08, 0.5, 0.03))
        self.slider_tv = Slider(ax_tv, "TV Loss", 0.0, 1000.0, valinit=self.lambda_tv)
        self.slider_tv.on_changed(self.set_lambda_tv)

        # --- lambda_distance ---
        ax_ld = plt.axes((0.4, 0.04, 0.5, 0.03))
        self.slider_dist_inner = Slider(ax_ld, "Inner Dist Loss", 0.0, 1000.0, valinit=self.lambda_distance_inner)
        self.slider_dist_inner.on_changed(self.set_lambda_distance_inner)

        ax_ld = plt.axes((0.4, 0.00, 0.5, 0.03))
        self.slider_dist_outer = Slider(ax_ld, "Outer Dist Loss", 0.0, 1000.0, valinit=self.lambda_distance_outer)
        self.slider_dist_outer.on_changed(self.set_lambda_distance_outer)

        # --- Class input ---
        ax_class = plt.axes((0.1, 0.02, 0.15, 0.05))
        self.class_box = TextBox(ax_class, "Class C")
        self.class_box.on_submit(self.set_class)

        # --- Run Optimization Button ---
        ax_button = plt.axes((0.1, 0.08, 0.15, 0.05))
        self.run_button = Button(ax_button, "Run")
        self.run_button.on_clicked(self.run_optimization)

        ax_load = plt.axes((0.1, 0.14, 0.15, 0.05))
        self.load_button = Button(ax_load, "Load Image")
        self.load_button.on_clicked(self.load_image)

        ax_toggle = plt.axes((0.1, 0.20, 0.15, 0.05))
        self.toggle_button = Button(ax_toggle, "Hide BBoxes")
        self.toggle_button.on_clicked(self.toggle_predictions)

    def load_path(self, file_path):
        pil_img = Image.open(file_path).convert("RGB")
        transform = T.Compose([T.Resize((self.height, self.width)), T.ToTensor()])
        self.x_tensor = transform(pil_img).unsqueeze(dim=0)

        self.update_image()
        self.region_selector.selected_region = None

    def load_image(self, event):
        if self.running:
            return

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.load_path(file_path)

    ###########################################################################
    # Widget callbacks
    ###########################################################################

    def set_class(self, text):
        self.class_C = text
        # print("Class C set to:", self.class_C)

    def set_iterations(self, val):
        self.num_iterations = int(val)
        # print("Iterations:", self.num_iterations)

    def set_lr(self, val):
        self.lr = float(val)
        # print("LR:", self.lr)

    def set_lambda_tv(self, val):
        self.lambda_tv = float(val)
        # print("TV Loss Weight:", self.lambda_tv)

    def set_lambda_distance_inner(self, val):
        self.lambda_distance_inner = float(val)
        # print("Inner Distance Loss Weight:", self.lambda_distance_inner)

    def set_lambda_distance_outer(self, val):
        self.lambda_distance_outer = float(val)
        # print("Outer Distance Loss Weight:", self.lambda_distance_outer)

    def toggle_predictions(self, event):
        if self.running:
            return

        self.show_predictions = not self.show_predictions
        self.toggle_button.label.set_text("Show BBoxes" if not self.show_predictions else "Hide BBoxes")
        self.update_image()

    def call_in_main_thread(self, func, *args, **kwargs):
        win = self.fig.canvas.manager.window
        win.after(0, lambda: func(*args, **kwargs))

    def optimize_image(self):
        print("RUNNING OPTIMIZATION")

        self.interp.model.model.eval()
        self.interp.set_seed_tensor(self.x_tensor)
        self.interp.optimizer.set_initial()

        self.interp.optimizer.params["REGION"] = list(map(int, list(self.region_selector.selected_region)))
        self.interp.optimizer.params["CLASS_ID"] = int(self.class_C)
        self.interp.optimizer.params["LAMBDA_DISTANCE_INNER"] = self.lambda_distance_inner
        self.interp.optimizer.params["LAMBDA_DISTANCE_OUTER"] = self.lambda_distance_outer
        self.interp.optimizer.params["LAMBDA_TV"] = self.lambda_tv

        self.progress_bar.set_width(0)

        def reset_bar():
            self.progress_bar.set_width(0)
            self.fig.canvas.draw_idle()

        self.call_in_main_thread(reset_bar)

        def progress_callback(i, total):
            progress = i / total

            def update_bar():
                self.progress_bar.set_width(progress)
                self.fig.canvas.draw_idle()

            # schedule bar update on main thread
            self.call_in_main_thread(update_bar)

        # Run heavy work in this background thread
        x_tensor_new = self.interp.optimizer.run(
            self.num_iterations,
            self.lr,
            progress_callback=progress_callback
        )

        # When finished, update state + image on main thread
        def finish():
            self.x_tensor = x_tensor_new
            self.running = False
            self.region_selector.disabled = False
            self.update_image()
            print("DONE RUNNING OPTIMIZATION")

        self.call_in_main_thread(finish)

    def run_optimization(self, event):
        if self.running:
            print("Currently Running")
            return

        if self.class_C is None:
            print("ERROR: No class C selected.")
            return

        if not self.class_C.isdigit():
            print("ERROR: Class C must be a positive integer.")
            return

        if int(self.class_C) >= 80:
            print("ERROR: Class C must be less than or equal to 80.")
            return

        self.region_selector.clear_rectangles()
        self.running = True
        self.region_selector.disabled = True

        region = self.region_selector.selected_region
        if region is None:
            print("ERROR: No region selected.")
            return

        threading.Thread(target=self.optimize_image).start()

    def draw_predictions(self):
        with torch.no_grad():
            results = self.yolo_pred(self.x_tensor.detach().clone(), conf=0.25, verbose=False)

        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().item()
            class_id = int(box.cls[0].cpu().item())
            label = f"{names[class_id]} {confidence:.4f}"

            # draw rectangle
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            self.ax.add_patch(rect)

            # draw label
            self.ax.text(
                x1,
                y1 - 5,
                label,
                color="red",
                fontsize=9,
                weight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="red",
                    boxstyle="round,pad=0.3"
                )
            )

    def update_image(self):
        self.ax.clear()
        self.ax.axis("off")

        image = get_np(self.x_tensor)
        self.ax.imshow(image)

        if self.show_predictions:
            self.draw_predictions()

        self.ax.figure.canvas.draw_idle()

    def start(self):
        plt.show()


if __name__ == "__main__":
    # For demonstration >>> replace with your seed image
    H, W = 320, 320

    gui = InterpretabilityGUI(H, W, "../images/road.jpg")
    gui.start()
