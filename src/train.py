import tensorflow as tf
from pathlib import Path
from hdf5storage import loadmat, savemat
import time
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import callbacks, regularizers
import tensorflow_probability as tfp
import os
from shutil import copyfile
from losses import L1, L2, Linf, Bias

# Turn off tensorflow verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():

    # U-net model parameters
    nfilt = 32  # Number of filters per convolution layer
    ksize = (3, 9)  # Size of filtering kernel
    nlayers = 3  # Number of convolution layers
    strides = 4  # Stride of max pooling
    lr = 1e-3  # Learning rate
    bsz = 4  # Batch size
    n_epochs = 500  # Number of training epochs
    reg2 = 1e-3  # L2 regularization of filter weights
    regtv = 1e-5  # Total variation regularization of output values
    kreg = regularizers.l2(reg2)  # Regularizer of filter kernel weights

    # Specify the data directory
    DATADIR = Path("../preprocess/data")

    # Apply 90/10 split for training/validation, and use the abdominal images for testing
    train_files = ["%s/aug/sample%04d.bin" % (DATADIR, i + 1) for i in range(900)]
    valid_files = ["%s/aug/sample%04d.bin" % (DATADIR, i + 1) for i in range(900, 1000)]
    test_files = ["%s/abd/sample%04d.bin" % (DATADIR, i + 1) for i in range(62)]

    # Load the image configuration
    mdict = loadmat("%s/aug/config.mat" % DATADIR)
    nx = mdict["thetas"].size
    nz = mdict["depths"].size
    nc = 1  # Number of output channels (1 for DAS data)
    t0 = mdict["thetas"].T.astype("float32")
    d0 = mdict["depths"].T.astype("float32")
    t1 = mdict["thetar"].T.astype("float32")
    d1 = mdict["depthr"].T.astype("float32")
    del mdict

    # Create data loader functions with augmentation for training, none for validation
    mb = strides**nlayers  # Minimum block size for U-net
    tloadfn = lambda b, nc=nc, nx=nx, nz=nz, mb=mb: load_flip(b, nc, nx, nz, mb)
    vloadfn = lambda b, nc=nc, nx=nx, nz=nz, mb=mb: load_data(b, nc, nx, nz, mb)
    trnload = lambda a: tf.vectorized_map(tloadfn, a)
    valload = lambda a: tf.vectorized_map(vloadfn, a)
    nz = (nz // mb) * mb
    t0 = t0[:nx, :]
    d0 = d0[:, :nz]

    # Create a TensorFlow dataset for the training and validation sets
    tds_f = tf.data.Dataset.list_files(train_files)
    vds_f = tf.data.Dataset.list_files(valid_files)
    par = {"num_parallel_calls": tf.data.AUTOTUNE}
    tds = tds_f.batch(bsz, **par).map(trnload, **par).prefetch(tf.data.AUTOTUNE)
    vds = vds_f.batch(bsz, **par).map(valload, **par).prefetch(tf.data.AUTOTUNE)

    # Prepare to run and store model (make directories, etc.)
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
    # Copy over the current script to the run directory
    copyfile(__file__, run_dir / Path(__file__).name)

    # Define inputs and outputs
    inputs = layers.Input(shape=(nc * 2, nx, nz), dtype=tf.float32, name="input")
    # outputs = make_model(inputs, nlayers=nlayers, nfilts=nfilt, ksize=ksize, kreg=kreg)
    outputs = make_unet(inputs, nlayers, nfilt, ksize, strides, kreg=kreg)

    # The training loss is MAE loss with total variation regularization
    def loss(p, y):
        return L1(p, y) + regtv * tf.reduce_mean(tf.image.total_variation(p))

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=loss,
        metrics=[L1, L2, Linf, Bias],
    )
    model.summary()

    # Add TensorBoard with B-mode image reconstruction
    (xt0, yt0), (xt1, yt1), (xt2, yt2) = [tloadfn(b) for b in train_files[:3]]
    (xv0, yv0), (xv1, yv1), (xv2, yv2) = [vloadfn(b) for b in valid_files[:3]]
    tdata = (tf.stack((xt0, xt1, xt2), 0), tf.stack((yt0, yt1, yt2), 0))
    vdata = (tf.stack((xv0, xv1, xv2), 0), tf.stack((yv0, yv1, yv2), 0))

    x = tf.stack((t1, d1), -1)
    x_ref_min = tf.stack((t0[+0, :], d0[:, +0]), -1)
    x_ref_max = tf.stack((t0[-1, :], d0[:, -1]), -1)

    tbCallBack = TensorBoardImage(
        tdata=tdata,
        vdata=vdata,
        x=x,
        x_ref_min=x_ref_min,
        x_ref_max=x_ref_max,
        log_dir=str(run_dir),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        profile_batch="10, 15",
    )

    # Train model
    history = model.fit(
        tds,
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
            "ksize": ksize,
            "nlayers": nlayers,
            "strides": strides,
            "lr": lr,
            "batch_size": bsz,
            "n_epochs": n_epochs,
            "reg2": reg2,
            "regtv": regtv,
        }
    )
    savemat(history_file, history.history)

    # Clear graph
    tf.keras.backend.clear_session()


def make_model(inputs, nlayers, nfilts, ksize, dr=[1000, 3000], kreg=None):
    """Simple CNN model with no pooling."""
    # Layer parameters
    par = {
        "padding": "same",
        "data_format": "channels_first",
        "kernel_regularizer": kreg,
    }

    x = inputs
    for _ in range(nlayers):
        x = layers.Conv2D(nfilts, ksize, activation="relu", **par)(x)
        x = layers.BatchNormalization(axis=1)(x)
    # Set the output of the last layer to have sigmoid activation
    x = layers.Conv2D(1, ksize, activation="sigmoid", **par)(x)
    # Rescale dynamic range from [0, 1] to dr
    x = x * (dr[1] - dr[0]) + dr[0]
    return x


def make_unet(inputs, nlayers, nfilts, ksize, strides=4, dr=[1000, 3000], kreg=None):
    """Basic U-net model, but without increasing the number of filters."""
    # Layer parameters
    par = {"padding": "same", "data_format": "channels_first"}
    par_conv = {**par, "activation": "relu", "kernel_regularizer": kreg}

    x = inputs
    # Encoder
    enc = []
    for _ in range(nlayers):
        x = layers.Conv2D(nfilts, ksize, **par_conv)(x)
        x = layers.BatchNormalization(axis=1)(x)
        enc.append(x)
        x = layers.MaxPooling2D(pool_size=(1, strides), **par)(x)
        # nfilts *= 2  # Double the number of filters
    # Decoder
    for i in range(nlayers):
        # nfilts //= 2  # Halve the number of filters
        x = layers.Conv2DTranspose(nfilts, ksize, (1, strides), **par_conv)(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Concatenate(axis=1)([x, enc[-(i + 1)]])

    # Set the output of the last layer to have sigmoid activation
    x = layers.Conv2D(1, ksize, activation="sigmoid", kernel_regularizer=kreg, **par)(x)
    # Rescale dynamic range from [0, 1] to dr
    x = x * (dr[1] - dr[0]) + dr[0]
    return x


@tf.function
def load_data(fname, nc, nx, nz, minblk):
    raw = tf.io.decode_raw(tf.io.read_file(fname), tf.float32)
    # The binary file has all of the input data followed by all of the target data
    x = raw[: 2 * nz * nx * nc]
    y = raw[2 * nz * nx * nc :]
    x = tf.reshape(x, (nc, nx, nz, 2))
    x = tf.transpose(x, (0, 3, 1, 2))  # Place real/imag in dim 1
    # x = x / (tf.norm(x, axis=1, keepdims=True) + 1e-20)
    x = tf.reshape(x, (nc * 2, nx, nz))  # Combine real/imag into channel dim
    y = tf.reshape(y, (1, nx, nz))
    # Make sure discretization is correct for the U-net
    # nx = (x.shape[-2] // minblk) * minblk
    nz = (x.shape[-1] // minblk) * minblk
    x = x[..., :nx, :nz]
    y = y[..., :nx, :nz]
    x = tf.where(tf.math.is_nan(x), 0.0 * x, x)
    return x, y


@tf.function
def load_flip(fname, nc, nx, nz, minblk=1):
    """Drop-in replacement for load_data with random left/right flipping."""
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

    def __init__(self, tdata, vdata, x, x_ref_min, x_ref_max, *args, **kwargs):
        # Use base class initialization
        self.tdata = tdata
        self.vdata = vdata
        # Interpolations
        self.x = tf.expand_dims(x, 0)
        self.x_ref_min = tf.expand_dims(x_ref_min, 0)
        self.x_ref_max = tf.expand_dims(x_ref_max, 0)
        super().__init__(*args, **kwargs)

    def convert_input(self, x):
        return tf.transpose((x[:, :1] + 1) / 2, (0, 3, 2, 1))

    def convert_output(self, p, dr=[1300, 2600]):
        c0 = (1540 - dr[0]) / (dr[1] - dr[0])
        p = (p - dr[0]) / (dr[1] - dr[0])
        p = tfp.math.batch_interp_regular_nd_grid(
            self.x, self.x_ref_min, self.x_ref_max, p, -2, c0
        )
        return tf.expand_dims(tf.transpose(p, (0, 2, 1)), -1)
        # return tf.transpose((p - dr[0]) / (dr[1] - dr[0]), (0, 3, 2, 1))

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


if __name__ == "__main__":
    main()
