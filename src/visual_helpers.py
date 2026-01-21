
import torch
from torchvision import transforms as T

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import utils


def visualize_result(x):
    optim_img = T.ToPILImage()(x.squeeze(0).detach().cpu())
    plt.imshow(optim_img)
    plt.axis('off')
    plt.title("Optimized Image")
    plt.show()


def visualize_result_w_bbs(model, x, conf_threshold=0.25, path=None, printing=True):
    img = T.ToPILImage()(x.squeeze(0).detach().cpu())
    results = model(x, conf=conf_threshold, verbose=False)

    if printing:
        print(f"\n{'=' * 60}")
        print(f"YOLO Predictions (confidence >= {conf_threshold}):")
        print(f"{'=' * 60}")

        if len(results[0].boxes) == 0:
            print("No detections found ):(")
        else:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())
                class_name = results[0].names[class_id]

                print(f"\nDetection {i + 1}:")
                print(f"  Class: {class_name} (ID: {class_id})")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  Center: ({(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f})")
                print(f"  Size: {x2 - x1:.1f} x {y2 - y1:.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(img)
    ax1.set_title('Optimized Image', fontsize=14)
    ax1.axis('off')

    ax2.imshow(img)
    ax2.set_title(f'YOLO Predictions (conf >= {conf_threshold})', fontsize=14)
    ax2.axis('off')

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().item()
        class_id = int(box.cls[0].cpu().item())
        class_name = results[0].names[class_id]

        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax2.add_patch(rect)

        # Add label
        label = f'{class_name} {confidence:.2f}'
        ax2.text(
            x1, y1 - 5, label,
            color='red', fontsize=10, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.3')
        )

    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
        plt.close()
        return

    plt.show()


def vis_tensor(tensor, in_channel=None, out_channel=None, nrow=8, padding=1):
    n, c, h, w = tensor.shape

    if out_channel is not None:
        tensor = tensor[out_channel, :, :, :].unsqueeze(0)

    if tensor.shape[1] != 3:
        if in_channel is None:
            raise RuntimeError("set specific channel to something since input channel size != 3")

        tensor = tensor[:, in_channel, :, :].unsqueeze(1)
        print(tensor.shape)

    rows = min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def visualize_layer_filters(weights):
    print(weights.shape)
    vis_tensor(weights, in_channel=2, out_channel=118)

