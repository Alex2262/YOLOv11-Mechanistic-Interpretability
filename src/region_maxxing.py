
# ******************************************************
# CRITICAL ASSUMPTION:
# H is divisible by 32 and W is divisible by 32
# I think if this assumption does not hold, then YOLO only works when H == W and it does some sort of padding
# but it is SIMPLEST if we proceed with this assumption
# ******************************************************


import math
import torch


STRIDES = [8, 16, 32]
REGION = [100, 100, 150, 180]

# REGION = [10, 10, 200, 200]
CLASS_ID = 0


def IoU(tx1, ty1, tx2, ty2, px1, py1, px2, py2):
    eps = 1e-9

    # -------------------------------
    # INTERSECTION
    # -------------------------------
    ix1 = torch.max(tx1, px1)
    iy1 = torch.max(ty1, py1)
    ix2 = torch.min(tx2, px2)
    iy2 = torch.min(ty2, py2)

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter_area = inter_w * inter_h

    # -------------------------------
    # UNION
    # -------------------------------
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    union = area_p + area_t - inter_area + eps

    # IoU
    iou = inter_area / union
    return iou


# generalized intersection over union
def GIoU(tx1, ty1, tx2, ty2, px1, py1, px2, py2):
    eps = 1e-9

    # -------------------------------
    # INTERSECTION
    # -------------------------------
    ix1 = torch.max(tx1, px1)
    iy1 = torch.max(ty1, py1)
    ix2 = torch.min(tx2, px2)
    iy2 = torch.min(ty2, py2)

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter_area = inter_w * inter_h

    # -------------------------------
    # UNION
    # -------------------------------
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    union = area_p + area_t - inter_area + eps

    # IoU
    iou = inter_area / union

    # -------------------------------
    # ENCLOSING BOX
    # -------------------------------
    cx1 = torch.min(tx1, px1)
    cy1 = torch.min(ty1, py1)
    cx2 = torch.max(tx2, px2)
    cy2 = torch.max(ty2, py2)

    enc_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps

    # -------------------------------
    # GIoU
    # -------------------------------
    giou = iou - (enc_area - union) / enc_area

    return giou


def bb_mse(tx1, ty1, tx2, ty2, px1, py1, px2, py2):
    return (tx1 - px1) ** 2 + (ty1 - py1) ** 2 + (tx2 - px2) ** 2 + (ty2 - py2) ** 2


def decode_cell(pred_box, head, grid_x, grid_y):
    pred_x, pred_y, pred_w, pred_h = pred_box

    x_center = pred_x
    y_center = pred_y
    w = pred_w
    h = pred_h

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    return torch.stack([x1, y1, x2, y2])


LAMBDA_CLASS = 100.0


def get_region_loss(inp, out, batch=0):
    # print(inp.size())

    _b, _c, h, w = inp.size()
    pred = out[0]

    x1, y1, x2, y2 = REGION

    tx1 = torch.tensor(REGION[0], dtype=torch.float32, device=pred.device)
    ty1 = torch.tensor(REGION[1], dtype=torch.float32, device=pred.device)
    tx2 = torch.tensor(REGION[2], dtype=torch.float32, device=pred.device)
    ty2 = torch.tensor(REGION[3], dtype=torch.float32, device=pred.device)

    assert (h % 32 == 0 and w % 32 == 0)
    assert (0 <= x1 < x2 < w and 0 <= y1 < y2 < h)

    # pred has shape: (1, 84, flattened_dim)
    # where flattened_dim sum of # of cells for all 3 heads
    # so:

    head_cells = [h // s * w // s for s in STRIDES]
    prefix_sum = [0]
    for s in head_cells:
        prefix_sum.append(s + prefix_sum[-1])

    flattened_dim = prefix_sum[3]
    assert (pred.size(2) == flattened_dim)

    rh = y2 - y1
    rw = x2 - x1
    rs = math.sqrt(rh * rw)

    # step 1: determine which heads to perform regression on
    if rs < 32:
        heads = [0]
    elif rs < 128:
        if CLASS_ID is None:
            heads = [0, 1]
        else:
            heads = [1]
    else:
        if CLASS_ID is None:
            heads = [0, 1, 2]
        else:
            heads = [2]

    loss = torch.tensor(0.0, dtype=torch.float32, device=inp.device)

    for head in heads:
        stride = STRIDES[head]
        offset = prefix_sum[head]

        grid_h = h // stride
        grid_w = w // stride

        # step 2: find all appropriate cells in region R
        # for a cell with x index grid_x, it covers x from [grid_x * stride, grid_x * stride + stride - 1]
        # for a cell with y index grid_y, it covers y from [grid_y * stride, grid_y * stride + stride - 1]

        # so:
        # to find grid_x_min we solve for grid_x_min * stride <= x1
        # so grid_x_min = floor((x1 / stride)) --> x1 // stride
        # to find grid_x_max we solve for grid_x_max * stride + stride - 1 >= x2
        # so grid_x_max = ceil((x2 + 1 - stride) / stride) --> (x2 + 1 - stride + stride - 1) // stride --> x2 // stride

        grid_x_min = x1 // stride
        grid_x_max = x2 // stride

        grid_y_min = y1 // stride
        grid_y_max = y2 // stride

        # print(grid_x_min, grid_y_min, grid_x_max, grid_y_max, head, stride)

        for grid_y in range(grid_y_min, grid_y_max + 1):
            for grid_x in range(grid_x_min, grid_x_max + 1):
                flattened_index = offset + grid_y * grid_w + grid_x
                pred_cell = pred[batch, :, flattened_index].to(inp.device)
                pred_box = pred_cell[0:4]
                pred_class = pred_cell[4:]

                # print(sum(pred_class))
                # print(max(pred_class), min(pred_class))

                if CLASS_ID is None:
                    class_loss = torch.sum(pred_class)
                    box_loss = 0
                else:
                    px1, py1, px2, py2 = decode_cell(pred_box, head, grid_x, grid_y)

                    # we can do box loss with 1 - GIoU
                    giou = GIoU(tx1, ty1, tx2, ty2, px1, py1, px2, py2)
                    box_loss = 1 - giou
                    target_class = torch.tensor(CLASS_ID, dtype=torch.long, device=pred.device)

                    class_loss = torch.nn.functional.cross_entropy(
                        pred_class.unsqueeze(0),  # Shape: (1, 80)
                        target_class.unsqueeze(0)  # Shape: (1,)
                    )

                loss += box_loss + LAMBDA_CLASS * class_loss

    return loss


def verify(inp, out, batch=0):
    _b, _c, h, w = inp.size()
    pred = out[0]

    x1, y1, x2, y2 = REGION

    assert (h % 32 == 0 and w % 32 == 0)
    assert (0 <= x1 < x2 < w and 0 <= y1 < y2 < h)

    head_cells = [h // s * w // s for s in STRIDES]
    prefix_sum = [0]
    for s in head_cells:
        prefix_sum.append(s + prefix_sum[-1])

    flattened_dim = prefix_sum[3]
    assert (pred.size(2) == flattened_dim)

    rh = y2 - y1
    rw = x2 - x1
    rs = math.sqrt(rh * rw)

    if rs < 32:
        heads = [0]
    elif rs < 128:
        if CLASS_ID is None:
            heads = [0, 1]
        else:
            heads = [1]
    else:
        if CLASS_ID is None:
            heads = [0, 1, 2]
        else:
            heads = [2]

    prob_s = 0
    prob_cnt = 0

    for head in heads:
        stride = STRIDES[head]
        offset = prefix_sum[head]

        grid_h = h // stride
        grid_w = w // stride

        grid_x_min = x1 // stride
        grid_x_max = x2 // stride

        grid_y_min = y1 // stride
        grid_y_max = y2 // stride

        for grid_y in range(grid_y_min, grid_y_max + 1):
            for grid_x in range(grid_x_min, grid_x_max + 1):
                flattened_index = offset + grid_y * grid_w + grid_x
                pred_cell = pred[batch, :, flattened_index].to(inp.device)
                pred_box = pred_cell[0:4]
                pred_class = pred_cell[4:]

                prob_s += pred_class[CLASS_ID]
                prob_cnt += 1

    return prob_s / prob_cnt


