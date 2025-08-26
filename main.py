from retinaface.model.retinaface_model import build_model


def main():
    model = build_model()
    model.summary()


if __name__ == "__main__":
    main()
