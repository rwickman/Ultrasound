from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

from model import DensityModel
from config import *
from dataset import create_datasets


model = DensityModel().eval()
model_dict = torch.load(os.path.join(save_dir, "model.pt"))
model.load_state_dict(model_dict["model"])


test_dataset = create_datasets(only_test=True)

test_loader = DataLoader(
    test_dataset,
    2,
    num_workers=2,
    prefetch_factor=4,
    drop_last = False,
    shuffle=False,
    multiprocessing_context="fork")

total_psnr = 0
total_ssim = 0
for batch in test_loader:
    SOS_true, FSA = batch
    # SOS_true = SOS_true.to(device)
    # FSA = FSA.to(device)
    SOS_pred = model(FSA)

    for i in range(SOS_pred.shape[0]):
        psnr = peak_signal_noise_ratio(SOS_pred[i,0].detach().numpy(),  SOS_true[i,0].detach().numpy())
        ssim = structural_similarity(SOS_pred[i,0].detach().numpy(),  SOS_true[i,0].detach().numpy())
        total_psnr += psnr
        total_ssim += ssim
        print("PSNR", psnr)
        print("SSIM", ssim)

print("Avg. PSNR: ", total_psnr / len(test_dataset))
print("Avg. SSIM: ", total_ssim / len(test_dataset))



