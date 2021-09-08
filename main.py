from model import net


def main():
    model = net()
    model.build_model()

    if model.opt.phase == "train":
        print("[*] Training begin!")
        model.train()
        print("[*] Training finished!")

    if model.opt.phase == "test":
        print("[*] Testing begin!")
        model.test()
        print("[*] Test finished!")


if __name__ == "__main__":
    main()
