import torch
import numpy as np
from config import cmin, cmax
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

L1 = lambda p, y: np.mean(np.abs(p - y))
L2 = lambda p, y: np.sqrt(np.mean(np.abs(p - y) ** 2))
Linf = lambda p, y: np.max(np.abs(p - y))
Bias = lambda p, y: np.mean(p - y)


class Metrics:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.reset()

    def reset(self):
        self.loss = 0
        self.ssim = 0
        self.psnr = 0
        self.l1 = 0
        self.l2 = 0
        self.linf = 0
        self.bias = 0
        self.count = 0

    def eval(self, cpred01, ctrue01):
        cpred01t = torch.tensor(cpred01).unsqueeze(1)
        ctrue01t = torch.tensor(ctrue01).unsqueeze(1)
        self.loss += np.array(self.loss_fn(cpred01t, ctrue01t).cpu())
        # cpred01t = torch.tensor(cpred01)
        # ctrue01t = torch.tensor(ctrue01)
        ctrue = ctrue01 * (cmax - cmin) + cmin
        cpred = cpred01 * (cmax - cmin) + cmin
        # self.loss += np.sum([self.loss_fn(p, y) for p, y in zip(cpred01t, ctrue01t)])
        self.ssim += np.sum([SSIM(p, y) for p, y in zip(cpred01, ctrue01)])
        self.psnr += np.sum([PSNR(p, y) for p, y in zip(cpred01, ctrue01)])
        self.l1 += np.sum([L1(p, y) for p, y in zip(cpred, ctrue)])
        self.l2 += np.sum([L2(p, y) for p, y in zip(cpred, ctrue)])
        self.linf += np.sum([Linf(p, y) for p, y in zip(cpred, ctrue)])
        self.bias += np.sum([Bias(p, y) for p, y in zip(cpred, ctrue)])
        self.count += len(cpred)

    def print(self, prefix):
        print(prefix + ":\t", end="")
        for key in ["loss", "ssim", "psnr", "l1", "l2", "linf", "bias"]:
            print(key + ": %0.4f" % (self.__dict__[key] / self.count), end="\t")
        print()

    def appendToDict(self, dict, var_prefix=""):
        for key in ["loss", "ssim", "psnr", "l1", "l2", "linf", "bias"]:
            if var_prefix + key not in dict:
                dict[var_prefix + key] = []
            dict[var_prefix + key].append(self.__dict__[key] / self.count)

    def gather(self):
        results = (
            self.loss / self.count,
            self.ssim / self.count,
            self.psnr / self.count,
            self.l1 / self.count,
            self.l2 / self.count,
            self.linf / self.count,
            self.bias / self.count,
        )
        return results
