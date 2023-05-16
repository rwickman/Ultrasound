import tensorflow as tf
import numpy as np
from pathlib import Path
from hdf5storage import loadmat, savemat
import time
from tensorflow.keras import (
    layers,
    Model,
    optimizers,
    losses,
    metrics,
    callbacks,
    regularizers,
)
import os
import re
from shutil import copyfile

# TensorBoard image parameters
# IMGIDX_TB = [0, 4, 8]  # Indexes of validation images to show in TensorBoard
# IMGIDX_TB = [0, 5, 10, 15, 20]  # Indexes of validation images to show in TensorBoard
# NIMGS_TB = len(IMGIDX_TB)  # Number of validation images to show in TensorBoard

# turn off tensorflow verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# DATADIR = Path("/home/dhyun/scratch/bathtub_data/data")
DATADIR = Path("../preprocess/data")

mdict = loadmat("%s/aug/config.mat" % DATADIR, variable_names=["nx", "nz", "nchout"])
nx = int(mdict["nx"][0][0])
nz = int(mdict["nz"][0][0])
nc = int(mdict["nchout"][0][0])
del mdict

# Parameters
nfilt = 32  # Number of filters per convolution layer
ksize = 11  # Size of filtering kernel
nlayers = 3  # Number of convolution layers
strides = 4  # Stride of max pooling
lr = 1e-3  # Learning rate
bsz = 8  # Batch size
n_epochs = 500  # Number of training epochs
# reg1 = 1e-3  # L1 regularization of filter weights
reg2 = 1e-3  # L2 regularization of filter weights
regtv = 1e-3  # Total variation regularization


train_files = ["%s/aug/sample%04d.bin" % (DATADIR, i + 1) for i in range(900)]
valid_files = ["%s/aug/sample%04d.bin" % (DATADIR, i + 1) for i in range(900, 1000)]
test_files = ["%s/abd/sample%04d.bin" % (DATADIR, i + 1) for i in range(62)]


# Load data into memory
minblk = strides**nlayers

tloadfn = lambda b, nc=nc, nx=nx, nz=nz, minblk=minblk: load_flip(b, nc, nx, nz, minblk)
vloadfn = lambda b, nc=nc, nx=nx, nz=nz, minblk=minblk: load_data(b, nc, nx, nz, minblk)
trnload = lambda a: tf.vectorized_map(tloadfn, a)
valload = lambda a: tf.vectorized_map(vloadfn, a)

nx = (nx // minblk) * minblk
nz = (nz // minblk) * minblk

def main():

    tds_f = tf.data.Dataset.list_files(train_files)
    vds_f = tf.data.Dataset.list_files(valid_files)

    par = {"num_parallel_calls": tf.data.AUTOTUNE}
    tds = tds_f.batch(bsz, **par).map(trnload, **par).prefetch(tf.data.AUTOTUNE)
    vds = vds_f.batch(bsz, **par).map(valload, **par).prefetch(tf.data.AUTOTUNE)

    # Prepare to run (make directories, etc.)
    run_name = time.strftime("%Y%m%d_%H%M")
    run_dir = Path("runs") / run_name
    model_file = str(run_dir / "model.h5")
    history_file = str(run_dir / "history.mat")
    print("Evaluating %s..." % run_dir)
    if not os.path.exists(run_dir):
        # If directory does not exist, create it
        os.makedirs(run_dir)
    else:
        # If directory exists and if complete, continue
        if os.path.exists(model_file) and os.path.exists(history_file):
            print("Output file already exists. Skipping...")
            return

    # Define inputs and outputs
    inputs = layers.Input(shape=(nc * 2, nx, nz), dtype=tf.float32, name="input")
    # outputs = make_model(inputs, nlayers=nlayers, nfilts=nfilt, ksize=ksize)
    outputs = make_unet(inputs, nlayers, nfilt, ksize, strides)

    # Define loss functions
    def L1(p, y):
        return tf.reduce_mean(tf.abs(p - y))

    def L2(p, y):
        return tf.sqrt(tf.reduce_mean(tf.abs(p - y) ** 2))

    def Linf(p, y):
        return tf.reduce_max(tf.abs(p - y))

    def Bias(p, y):
        return tf.reduce_mean(p - y)

    # MAE loss with total variation regularization
    def loss(p, y):
        # p = tf.math.log1p(p)
        # y = tf.math.log1p(y)
        return L1(p, y) + regtv * tf.reduce_mean(tf.image.total_variation(p))

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=loss,
        metrics=[L1, L2, Linf, Bias],
    )

    # Add TensorBoard with B-mode image reconstruction
    (xt0, yt0), (xt1, yt1), (xt2, yt2) = [tloadfn(b) for b in train_files[:3]]
    (xv0, yv0), (xv1, yv1), (xv2, yv2) = [vloadfn(b) for b in valid_files[:3]]
    tdata = (tf.stack((xt0, xt1, xt2), 0), tf.stack((yt0, yt1, yt2), 0))
    vdata = (tf.stack((xv0, xv1, xv2), 0), tf.stack((yv0, yv1, yv2), 0))
    tbCallBack = TensorBoardImage(
        tdata=tdata,
        vdata=vdata,
        log_dir=str(run_dir),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        profile_batch="10, 15",
    )

    # Train model
    history = model.fit(
        tds,
        # batch_size=bsz,
        epochs=n_epochs,
        validation_data=vds,
        verbose=1,
        shuffle=True,
        callbacks=[tbCallBack],
    )

    # Save model
    model.save_weights(model_file, save_format="h5")

    # Save history plus hyperparameters
    history.history.update(
        {
            "nfilt": nfilt,
            "lr": lr,
            "batch_size": bsz,
            "n_epochs": n_epochs,
            # "reg1": reg1,
            # "reg2": reg2,
            "regtv": regtv,
        }
    )
    savemat(history_file, history.history)

    # Clear graph
    tf.keras.backend.clear_session()


def make_model(inputs, nlayers, nfilts, ksize, dr=[1000, 3000]):
    x = inputs
    params = {
        "padding": "same",
        "data_format": "channels_first",
        "kernel_regularizer": regularizers.l2(reg2),
    }

    for _ in range(nlayers):
        x = layers.Conv2D(nfilts, ksize, activation="relu", **params)(x)
        x = layers.BatchNormalization(axis=1)(x)
    x = layers.Conv2D(1, ksize, activation="sigmoid", **params)(x)
    x = x * (dr[1] - dr[0]) + dr[0]
    return x


def make_unet(inputs, nlayers, nfilts, ksize, strides=4, dr=[1000, 3000]):
    x = inputs
    cpar = {
        "padding": "same",
        "data_format": "channels_first",
        "activation": "relu",
        "kernel_regularizer": regularizers.l2(reg2),
    }
    ppar = {"padding": "same", "data_format": "channels_first"}
    # Encoder
    enc = []
    for _ in range(nlayers):
        x = layers.Conv2D(nfilts, ksize, **cpar)(x)
        x = layers.BatchNormalization(axis=1)(x)
        enc.append(x)
        x = layers.MaxPooling2D(pool_size=strides, **ppar)(x)
        # nfilts *= 2
    # Decoder
    for i in range(nlayers):
        # nfilts //= 2
        x = layers.Conv2DTranspose(nfilts, ksize, strides, **cpar)(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Concatenate(axis=1)([x, enc[-(i + 1)]])

    # Output
    x = layers.Conv2D(
        1, ksize, activation="sigmoid", kernel_regularizer=regularizers.l2(reg2), **ppar
    )(x)

    x = x * (dr[1] - dr[0]) + dr[0]
    return x


# @tf.function
def load_data(fname, nc, nx, nz, minblk):
    raw = tf.io.decode_raw(tf.io.read_file(fname), tf.float32)
    x = raw[: 2 * nz * nx * nc]
    y = raw[2 * nz * nx * nc :]
    x = tf.reshape(x, (nc, nx, nz, 2))
    x = tf.transpose(x, (0, 3, 1, 2))  # Place real/imag in dim 1
    x = x / tf.norm(x, axis=1, keepdims=True)
    x = tf.reshape(x, (nc * 2, nx, nz))  # Combine real/imag into channel dim
    y = tf.reshape(y, (1, nx, nz))
    # Make sure discretization is correct
    nx = (x.shape[-2] // minblk) * minblk
    nz = (x.shape[-1] // minblk) * minblk
    x = x[..., :nx, :nz]
    y = y[..., :nx, :nz]
    return x, y


@tf.function
def load_flip(fname, nc, nx, nz, minblk=1):
    x, y = load_data(fname, nc, nx, nz, minblk)
    sz = x.shape
    nc, nx, nz = sz[0] // 2, sz[1], sz[2]
    # Add augmentation
    if tf.random.uniform(()) < 0.5:
        x = tf.reshape(x, (nc, 2, nx, nz))
        x = tf.reverse(x, axis=(0, 2))
        y = tf.reverse(y, axis=(1,))
        x = tf.reshape(x, (nc * 2, nx, nz))  # Combine real/imag into channel dim
    return x, y


class TensorBoardImage(callbacks.TensorBoard):
    """
    TensorBoardImage extends tf.keras.callbacks.TensorBoard, adding custom processing
    upon setup and after every epoch to store properly processed ultrasound images.
    """

    def __init__(self, tdata, vdata, *args, **kwargs):
        # Use base class initialization
        self.tdata = tdata
        self.vdata = vdata
        super().__init__(*args, **kwargs)

    def convert_input(self, x):
        return tf.transpose((x[:, :1] + 1) / 2, (0, 3, 2, 1))

    def convert_output(self, p, dr=[1300, 2600]):
        return tf.transpose((p - dr[0]) / (dr[1] - dr[0]), (0, 3, 2, 1))

    def on_epoch_end(self, epoch, logs={}):
        """At the end of each epoch, add prediction images to TensorBoard."""
        # Use base class implementation
        super().on_epoch_end(epoch, logs)

        if epoch % 10 == 0:
            writer = tf.summary.create_file_writer("%s/images" % self.log_dir)
            with writer.as_default():
                xtrn = self.tdata[0]
                ytrn = self.tdata[1]
                ptrn = self.model(xtrn)
                p = self.convert_output(ptrn)
                tf.summary.image("T Output", p, step=epoch)
                if epoch == 0:
                    x = self.convert_input(xtrn)
                    y = self.convert_output(ytrn)
                    tf.summary.image("T Input", x, step=epoch)
                    tf.summary.image("T Target", y, step=epoch)

                xval = self.vdata[0]
                yval = self.vdata[1]
                pval = self.model(xval)
                p = self.convert_output(pval)
                tf.summary.image("V Output", p, step=epoch)
                if epoch == 0:
                    x = self.convert_input(xval)
                    y = self.convert_output(yval)
                    tf.summary.image("V Input", x, step=epoch)
                    tf.summary.image("V Target", y, step=epoch)
                tf.summary.flush()

            # # Get validation data
            # xval = self.val_data[0][IMGIDX_TB]
            # # yval = self.val_data[1][IMGIDX_TB]

            # # Add the predicted image summary to TensorBoard every epoch
            # feed_dict = {self.model.inputs[0]: xval}
            # # feed_dict = {self.model.inputs[0]: xval, self.model.targets[0]: yval}
            # ps = tf.keras.backend.get_session().run(self.psumm, feed_dict=feed_dict)
            # # xs, ys, ps = tf.keras.backend.get_session().run(
            # xs, ps = tf.keras.backend.get_session().run(
            #     [self.xsumm, self.psumm],
            #     feed_dict=feed_dict
            #     # [self.xsumm, self.ysumm, self.psumm], feed_dict=feed_dict
            # )
            # self.writer.add_summary(ps, epoch)

        # # Add the input and target summary to TensorBoard only on first epoch
        # if epoch == 0:
        #     self.writer.add_summary(xs, 0)
        #     # self.writer.add_summary(ys, 0)

        # self.writer.flush()


if __name__ == "__main__":
    print("Running main.")
    main()
    print("Ran main.")
