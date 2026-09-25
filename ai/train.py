from .training import train


if __name__ == "__main__":

    train(
        epochs=30,
        batch_size=64,
        learning_rate=0.001
    )