import keras


class DiceLoss(keras.losses.Loss):
    """Custom Dice Loss function for semantic segmentation.
    
    This loss function calculates the Dice coefficient, a similarity measure
    commonly used in semantic segmentation tasks. It penalizes false positives
    and false negatives, aiming to maximize the overlap between the predicted
    and ground truth segmentation masks.
    
    Args:
        name (str): Optional name for the loss function.
        alpha (float): Smoothing term to avoid division by zero.
    """
    def __init__(self, name="dice_loss", alpha=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        y_pred = keras.activations.sigmoid(y_pred)
        intersection = keras.ops.sum(y_true * y_pred, axis=-1)
        union = keras.ops.sum(y_true, axis=-1) + keras.ops.sum(y_pred, axis=-1) 
        dice = (2.0 * intersection + self.alpha) / (union + self.alpha)
        return 1 - dice
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "alpha": self.alpha}


class MixedLoss(keras.losses.Loss):
    """
    Mixed Loss function combining Dice Loss and Binary Focal Loss.
    
    This loss function combines the Dice Loss and Binary Focal Loss to address
    class imbalance and pixel-wise segmentation accuracy in semantic segmentation
    tasks. The Dice Loss measures the overlap between predicted and ground truth
    masks, while the Binary Focal Loss penalizes false positives and false negatives.

    Args:
        alpha (float): Weighting factor for the focal loss.
        gamma (float): Focusing parameter for the Binary Focal Loss.
        epsilon (float): Smoothing term to avoid numerical instability.
        name (str): Optional name for the loss function.

    Attributes:
        alpha (float): Weighting factor for the focal loss.
        gamma (float): Focusing parameter for the Binary Focal Loss.
        epsilon (float): Smoothing term used in the logarithm calculation.
        dice_loss (DiceLoss): Instance of the Dice Loss function.
        focal_loss (BinaryFocalCrossentropy): Instance of the Binary Focal Loss function.
    """
    def __init__(self, alpha=10.0, gamma=2.0, epsilon=1e-7, name="mixed_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  
        self.dice_loss = DiceLoss()
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(gamma=gamma)
        
    def call(self, y_true, y_pred):
        y_pred = keras.activations.sigmoid(y_pred)
        dice_loss = self.dice_loss(y_true, y_pred)
        focal_loss = self.focal_loss(y_true, y_pred)
        total_loss = self.alpha * focal_loss - keras.ops.log(dice_loss + self.epsilon) 
        return keras.ops.mean(total_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "alpha": self.alpha,
               "gamma": self.gamma, "epsilon": self.epsilon}