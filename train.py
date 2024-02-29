import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import utils, metrics, losses 


## Basic parameters
BASE_DIR = ""
# images directory
TRAIN_DIR = os.path.join(BASE_DIR, "data/train_v2")
# target image segmentations file
DF_PATH = os.path.join(BASE_DIR, "data/train_ship_segmentations_v2.csv")
# training hyperparameters
BATCH_SIZE = 64
EPOCHS = 10


## Data Preparation
# load target segmentations file
segmentations = pd.read_csv(DF_PATH)
# Group RLEs by 'ImageId' and aggregate them into lists
grouped = segmentations.groupby('ImageId')['EncodedPixels'].agg(agg_rles).to_frame().reset_index()
# cpunt the number of target objects for each image
grouped["counts"] = grouped["EncodedPixels"].apply(lambda x: x if x == 0 else len(x))
# mask to discard images with no target
null_mask = grouped["EncodedPixels"] == 0
# split images with targets into training and validation sets, excluding images with 0 targets to segment
train_ids, valid_ids = train_test_split(grouped[~null_mask], test_size=0.1, random_state=42)

# split into image paths and their corresponding RLEs
train_paths, train_rles = utils.get_image_path_rle(train_ids)
valid_paths, valid_rles = utils.get_image_path_rle(valid_ids)

# get batched training set
train_ds = utils.get_dataset(train_paths, train_rles, patch_size=256, batch_size=BATCH_SIZE)
train_ds = train_ds.map(
    utils.Augment(), 
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE).repeat()

# prepare validation set normally
valid_ds = utils.get_dataset(valid_paths, valid_rles, patch_size=256, batch_size=BATCH_SIZE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

## Model Architecture
@keras.utils.register_keras_serializable()
class Upsample(keras.layers.Layer):
    """
    Upsampling layer for the UNetMobile model.

    This layer performs convolutional transpose followed by batch normalization,
    dropout (optional), and ReLU activation.

    Args:
        filters (int): Number of filters in Conv2DTranspose.
        size (int or tuple of int): Size of the Conv2DTranspose convolutional kernel.
        name (str): Optional name for the layer.
        use_dropout (bool): Whether to apply dropout. Defaults to False.
    """
    def __init__(self, filters, size, name, use_dropout=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.size = size
        self.use_dropout = use_dropout
        self.conv_transpose = layers.Conv2DTranspose(
            filters=self.filters, kernel_size=self.size, strides=2, 
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            use_bias=False, padding="same")
        self.batch_norm = layers.BatchNormalization()
        if self.use_dropout:
            self.dropout = layers.Dropout(rate=0.5)
        self.relu = layers.ReLU()
   
    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.batch_norm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x
        
    def get_config(self):
        return dict(filters=self.filters, size=self.size, 
                    use_dropout=self.use_dropout, **super().get_config())

@keras.saving.register_keras_serializable()
class UNetMobile(keras.Model):
    """
    UNetMobile model for semantic segmentation.

    This model extends MobileNetV2 and adds upsampling layers for segmentation.

    Args:
        input_shape (list): Input shape of the model. Defaults to [224, 224, 3].
        output_channels (int): Number of output masks (classes to predict). Defaults to 1.
        name (str): Optional name for the model.
    """
    def __init__(self, input_shape=[224, 224, 3], output_channels=1, name="unet_mobile", **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self._input_shape = input_shape
        self.layer_to_reuse = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        self.base_model = keras.applications.MobileNetV2(input_shape=input_shape, weights="imagenet", include_top=False)
        self.base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_to_reuse]
        # Image downsampling path
        self.down_stack = keras.Model(inputs=self.base_model.input, outputs=self.base_model_outputs)
        
        # Freeze the weights of the base model
        self.down_stack.trainable = False
        
        # Define upsampling layers
        self.n_filters = [512, 256, 128, 64]
        params = [(f, 3, f"upsample_{f}") for f in self.n_filters]
        self.up_stack = [Upsample(*p_set) for p_set in params]
        
        # Define the last convolutional layer
        self.last_conv = layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same', dtype='float32')

            
    def call(self, inputs):
        # Connect downsampling and upsampling layers into a model
        x = self.down_stack(inputs)
        skips = reversed(x[:-1])
        x = x[-1]
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = layers.Concatenate()
            x = concat([x, skip])
        x = self.last_conv(x)
        return x
    
    def summary(self):
        # Get model summary along with output shapes and number of parameters
        x = keras.Input(shape=(224, 224, 3))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
    
    def get_config(self):
        return dict(
            input_shape=self._input_shape,
            output_channels=self.output_channels,
            **super().get_config())        


## Model pretraining
# the optimal learning rate was found using log-LR/loss plot 
# define and compile the model 
model = UNetMobile()
model.compile(loss=losses.MixedLoss(), optimizer=Adam(learning_rate=1e-2), metrics=[metrics.dice_score])

# set up training callbacks
model_path = "models/model.keras"

checkpoint = keras.callbacks.ModelCheckpoint(
    model_path, monitor='val_dice_score', 
    verbose=1, save_best_only=True, mode='max')

reduce_on_pl = keras.callbacks.ReduceLROnPlateau(
    monitor="val_dice_score", factor=0.2, patience=1, 
    verbose=1, mode='max', epsilon=0.001, min_lr=1e-3
)

e_stop = EarlyStopping(monitor="val_dice_score", mode="max", patience=3)
# all callbacks for model pretraining
callbacks = [checkpoint, e_stop, reduce_on_pl]
# training hyperparameters
steps_per_epoch = len(train_paths) // BATCH_SIZE 
# train the model 
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=valid_ds,
    batch_size=BATCH_SIZE,
    callbacks=callbacks)


## Model Fine-Tuning
# model is fine-tuned on full images, without cropping, only resizing and rescaling to [-1, 1]
train_ds = utils.get_dataset(train_paths, train_rles, patch_size=None, batch_size=BATCH_SIZE)
train_ds = train_ds.map(
    utils.Augment(), 
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE).repeat()

valid_ds = utils.get_dataset(valid_paths, valid_rles, patch_size=None, batch_size=BATCH_SIZE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# unfreeze downsampling layers except for BatchNormalization
model.layers[1].trainable = True
for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
        
# lower learning rate and compile       
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss=losses.MixedLoss(), optimizer=optimizer, metrics=[metrics.dice_score])

# update learning rate scheduling factor, min_lr based on learning rate finder plot 
reduce_on_pl = keras.callbacks.ReduceLROnPlateau(
    monitor="val_dice_score", factor=0.5, patience=1, 
    verbose=1, mode='max', epsilon=0.001, min_lr=5e-5
)

# update callbacks list
callbacks = [checkpoint, e_stop, reduce_on_pl]

# fine-tuning
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=valid_ds,
    batch_size=batch_size,
    callbacks=callbacks,
)