import keras
import numpy as np
import tensorflow as tf


def get_image_path_rle(df, target_dir=DF_PATH):
    """Extracts image paths and corresponding RLE-encoded masks from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image filenames and RLE-encoded masks.
        target_dir (str): Directory containing the images. Defaults to train_dir.

    Returns:
        tuple: A tuple containing a list of image paths and a list of RLE-encoded masks.
    """
    paths = [os.path.join(target_dir, img_name) for img_name in df.iloc[:, 0]]
    rles = df.loc[:, "EncodedPixels"].tolist()
    filtered_rles = [rle if rle else [] for rle in rles]
    return paths, filtered_rles


def rle_to_mask(rle_list, img_shape):
    """
    Calculates image mask based on its RLEs.
    
    Parameters:
    rle_list (string): image RLEs;
    img_shape (list): shape of input image.
    
    Returns: 
    array: image mask of input image shape,
    where 0 (background), 1 (target object). 
    """
    # Initialize a mask as an array full of zeros
    mask = np.zeros((img_shape[0] * img_shape[1]), dtype=np.uint8)
    if len(rle_list) > 0:
        # Iterate over each RLE string in the nested list
        for rle in rle_list:
            # Split the RLE string into individual elements
            rle_list = [int(x) for x in rle.split()]
            # Update mask values based on values extracted from the RLE string
            for i in range(0, len(rle_list), 2):
                # Extract start pixel and subtract 1 to ensure 0-indexing
                start_pixel = rle_list[i] - 1
                # Extract length value
                num_pixels = rle_list[i + 1]
                # Set the corresponding values of mask to 1s 
                mask[start_pixel:start_pixel + num_pixels] = 1
    # Reshape the mask according to the size of the original image
    mask = mask.reshape((img_shape[0], img_shape[1])).T
    # add channel dimension
    return np.expand_dims(mask, axis=-1)


def crop_patch(image, mask, patch_size=256):
    """
    Crops symmetric patches from the input image and its mask centered around an object in the mask,
    if it exists. If no object is found in the mask, a random patch of the same size is cropped.

    Parameters:
    image (array): The input image.
    mask (array): The mask indicating the location of objects in the image.
    patch_size (int, optional): The size of the square patch to be cropped. Default is 256.

    Returns:
    tuple: A tuple containing the cropped image and its corresponding mask.
    """
    # Check if an object exists
    if tf.reduce_any(mask):
        # Find indices of object pixels in the mask
        object_indices = tf.where(tf.equal(mask, 1))
        # Select a pair of indices randomly
        selected_index = tf.random.shuffle(object_indices)[0]
        # Determine the grid cell containing the selected pixel
        grid_id = (selected_index // patch_size) * patch_size
        # Set up cropping indices
        ind = [
            grid_id[0], grid_id[0] + patch_size,
            grid_id[1], grid_id[1] + patch_size
        ]
        # Crop image and its mask based on selected indices
        cropped_image = image[ind[0]:ind[1], ind[2]:ind[3], :]
        cropped_mask = mask[ind[0]:ind[1], ind[2]:ind[3], :]
    else:
        # Crop random patch from image and its mask
        cropped_image = tf.image.random_crop(image, [patch_size, patch_size, 3])
        cropped_mask = mask[:patch_size, :patch_size, :] 
    # Cropped (image, mask) pair
    return cropped_image, cropped_mask


def single_example(row, patch_size, size=[224, 224]):
    """
    Processes a single example for input into the model.

    This function reads an image file from the specified DataFrame row, decodes it, converts it to floating-point format,
    converts the mask from RLE format to a binary mask, crops the image and mask if a patch size is specified,
    and resizes both the image and mask to a standardized size.

    Parameters:
    row (tuple): A tuple containing information about the image file and its corresponding mask.
                 The first element is the file path of the image, and the second element is the RLE-encoded mask.
    patch_size (int): The size of the square patch to be cropped. If set to None, no cropping is performed.

    Returns:
    tuple: A tuple containing the processed input image and its corresponding mask.
    """
    # Read image file from row and decode it
    input_img = tf.io.read_file(row[0])
    input_img = tf.io.decode_jpeg(input_img)
    # Convert the image to float32 data type
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    # Create mask for a given image
    mask = rle_to_mask(row[1], input_img.shape)
    # If specified, crop patch from image and mask
    if patch_size:
        input_img, mask = crop_patch(input_img, mask, patch_size=patch_size)
    # Resize the image and mask to the specified size
    input_img, mask = resize_rescale(input_img, mask, size)
    # return preprocessed image and mask
    return input_img, mask


def resize_rescale(image, mask, size):
    """
    Resizes the image and mask to the specified size and rescales the image to the range [-1, 1].

    Parameters:
        image (Tensor): The input image tensor.
        mask (Tensor): The input mask tensor.
        size (tuple): The target size (height, width) to resize the image and mask.

    Returns:
        tuple: A tuple containing the resized and rescaled image and mask.
    """
    image = tf.image.resize(image, size) 
    mask = tf.image.resize(mask, size)
    image = image * 255 / 127.5 - 1
    return image, mask


def get_dataset(paths, rles, patch_size=256, batch_size=32):
    """
    Creates a TensorFlow dataset from image paths and RLE-encoded masks.

    Parameters:
        paths (list): A list of file paths to the images.
        rles (list): A list of RLE-encoded masks.
        patch_size (int): The size of the patches to crop.
        batch_size (int): The batch size for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    # Define a generator function to generate dataset elements
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(paths, rles, patch_size=patch_size),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        )
    )
    return dataset.batch(batch_size, drop_remainder=True)


def get_test_image(path, size):
    """
    Reads an image file, resizes it, and scales its values to the range [-1, 1].

    Parameters:
        path (str): The path to the image file.
        size (list): The target size of the image.

    Returns:
        tf.Tensor: The processed image tensor.
    """
    input_img = tf.io.read_file(path)
    input_img = tf.io.decode_jpeg(input_img)
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    input_img = tf.image.resize(input_img, size) * 255 / 127.5 - 1
    return input_img


def get_test_dataset(directory, size=[224, 224]):
    """
    Creates a TensorFlow dataset from image files in a directory for testing.

    Parameters:
        directory (str): The path to the directory containing the image files.
        size (list): The target size of the images.

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    dataset = tf.data.Dataset.list_files(str(directory + "*"))
    dataset = dataset.map(lambda x: get_test_image(x, size))
    return dataset


class Augment(keras.layers.Layer):
    def __init__(self, bright_factor=0.2, seed=42, **kwargs):
        """
        Initializes the Augment layer.

        Parameters:
            bright_factor (float): The factor for adjusting brightness.
            seed (int): The seed value for random augmentation.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.image_h_flip = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.mask_h_flip = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.image_v_flip = keras.layers.RandomFlip(mode="vertical", seed=seed)
        self.mask_v_flip = keras.layers.RandomFlip(mode="vertical", seed=seed)
        self.bright = keras.layers.RandomBrightness(bright_factor, value_range=(-1, 1))

    def call(self, image, mask):
        """
        Applies data augmentation to the input image and mask.

        Parameters:
            image (tf.Tensor): The input image tensor.
            mask (tf.Tensor): The input mask tensor.

        Returns:
            tuple: A tuple containing the augmented image and mask tensors.
        """
        image, mask = self.image_h_flip(image), self.mask_h_flip(mask)
        image, mask = self.image_v_flip(image), self.mask_v_flip(mask)
        image = self.bright(image)
        return (image, mask)


def predict_mask(img_path, model, threshold=0.4):
    """
    Get single prediction from model given full path to the image
    Returns: (image, predicted mask)
    """
    # load image
    image = tf.keras.utils.load_img(img_path)
    # convert to array
    image_arr = tf.keras.utils.img_to_array(image)
    # get shape
    image_shape = image_arr.shape
    # resize to the shape required by the model
    rescaled = tf.image.resize(image_arr, [224, 224]) / 127.5 - 1
    # predict mask for the image
    pred_mask = model.predict(tf.expand_dims(rescaled, axis=0), verbose=0)
    # apply thresholding
    pred_mask = tf.cast(pred_mask > threshold, tf.int32)
    # resize mask to the image original size
    pred_mask = tf.image.resize(pred_mask, [image_shape[0], image_shape[1]])
    # remove batch dimension
    pred_mask = tf.squeeze(pred_mask)
    return image, pred_mask

def get_batch_preds(paths, model):
    """
    Get predictions for batch of images given list of paths to images and a model
    Returns: list of (image, mask) pairs
    """
    preds = []
    for path in paths:
        preds.append(predict_mask(path, model))
    return preds