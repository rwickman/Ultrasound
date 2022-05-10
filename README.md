
# Setup
Install the package and dependencies:
```shell
pip install .
```

In order to train faster, I converted the .mat input files into .npy files. You will need to first copy `create_npy.py` to the directory of the datasets and then run:
```shell
python create_npy.py
```

All the necessary configurations for this project can be found `ultrasound/train/config.py`.

# Training
```shell
python ultrasound/train/train.py
```

To plot the training figures, first change the `model_dir` variable in `ultrasound/util/plot_results` and then run:
```shell
python ultrasound/util/plot_results.py
```

To show results of the current model run:
```shell
python ultrasound/util/plot_validation.py
```
This currently plots the results on the training dataset. To change it to a different set, change argument `train_dataset` given to `val_loader` to `val_dataset` or `test_dataset`.

# Test the model
To get the average SSIM and PSNR scores on the test datset run:
```shell
python ultrasound/train/test.py
```
