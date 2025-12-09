

from yolo_interp import YoloInterp
from auto import Auto


def main():
    y = YoloInterp(device="cpu")

    y.print_layers()

    # y.set_seed("images/road.jpg")
    # y.targets.set_conv_layer(8, None)
    # y.optimizer.run(1000, 0.01)

    # y.set_seed("images/road.jpg")
    # y.targets.set_conv_layer(5, [118,  62,  35,  27,  98, 114,  51,  95,  71,  22])
    # y.optimizer.run(1000, 0.005)

    a = Auto(3)
    # a.auto_layer([17])
    # a.auto_layer([3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15])

    # a.bin(5, 27)
    a.bin_layer(5)


if __name__ == '__main__':
    main()
