import numpy as np
from train import Trainer
from hyperopt import hp, fmin, tpe

counter = 0


def run(lr):
    global counter
    counter += 1

    trainer = Trainer("lrtune_%03d" % counter)
    return trainer.train(lr, False)


def main():
    # Define a search space
    space = hp.loguniform("lr", np.log(10**-6), np.log(10**-2))
    best = fmin(lambda lr: run(lr), space, algo=tpe.suggest, max_evals=50)
    print("Wow!")


if __name__ == "__main__":
    main()
