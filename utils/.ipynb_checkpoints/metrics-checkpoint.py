import keras

def dice_score(y_true, y_pred):
    """"
    Computes the Dice similarity coefficient metric for semantic segmentation.
    
    The Dice coefficient measures the similarity between predicted and ground
    truth masks. It is defined as twice the intersection of the predicted and
    ground truth masks divided by the sum of their sizes.
    """
    y_pred = keras.ops.cast(y_pred > 0, y_true.dtype)
    intersection = keras.ops.sum(y_true * y_pred)
    union = keras.ops.sum(y_true) + keras.ops.sum(y_pred)
    score = (2.0 * intersection) / (union + 1.0)
    return score