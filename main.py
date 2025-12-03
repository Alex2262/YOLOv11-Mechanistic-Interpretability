

from yolo_interp import YoloInterp


def main():
    y = YoloInterp(device="cpu")

    y.print_layers()

    y.set_seed("images/dog.jpg")
    y.targets.set_conv0(2)
    y.optimizer.run(10, 0.1)


if __name__ == '__main__':
    main()
