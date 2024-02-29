This repository is dedicated to the semantic segmentation model training for Airbus Ship Detection Challenge hosted by Kaggle:
- Task: Semantic Segmentation
- Data: https://www.kaggle.com/competitions/airbus-ship-detection/data
- Objective: trained semantic segmentation model


Required instrumets: requirements.txt

Repository structure overview:
1. data (training data (none), testing data (a few testing images))
2. models (saved .keras trained model)
3. notebooks (data analysis notebook)
4. utils (contains losses, metrics and other helper functions)
5. train.py (model training script)
6. test.py (inference script)


The process of tackling this problem consists of several substeps:

1. Problem domain investigation for:
   - existing solutions (could serve as a baseline model for further work and performance comparison);
   - latest research in the area (possible guidlines for optimal solutions space elimination) [1];


2. Data analysis:
   - data structure:
     labels: images of size 768x768;
     targets: image masks that are stored in numerical form in a file train_ship_segmentations_v2.csv; 
   - data overview:
     The preliminary data overview showed strong data imbalance, where the number of images with no targets to segment (150000 images (78.0% of the total data)) is substantially higher than images that contain targets (42556 images (22.0%)). The total number of data entries is 231723, where the number of unique images is 192556, which means that some images contain multiple objects to segment. Moreover, the targets themselves most of the time do not take up more than 1-2% of the total image pixels. Problem domain investigation results suggested to discard all the "blank" images and continue with only "target-containing" ones. This allows significantly decrease training time and balances the training data. The pixel imbalance can be alleviated by subsequent image transformations, for example using cropping.


3. Data Preprocessing:
   - preliminary source preprocessing:
      The image paths with the corresponding mask encodings (RLEs) are stored in a train_ship_segmentations_v2.csv which is read into a pandas DataFrame. Then the dataframe is transformed by grouping RLEs by "ImageId" which are aggregated into lists of RLEs, where one row of data corresponds to the following information:
        1. "ImageId" (unique image name);
        2. "EncodedPixels": list of lists of encoded target objects (if image contains any), "0" if there is no target objects.
   - training / validation splits generation:
      Splits were created based on grouped DataFrame "ImageId" column using sklearns train_test_split() function using 90/10 split for training and validation set correspondingly.
  
   - data preprocessing:
Due to different formats in which the input data stored, separate preprocessing logic must be incorporated to load and process both images and label and then combine the pairs into training instances. To solve this task tf.data.Dataset.from_generator() is used with custom function to read and prepare data on the fly. First transformed DataFrame is split by columns into 2 lists: image paths (for each image id the full path where image is located is created) and RLEs list of lists. Then, for each (image path, rle list) generator creates a training instance by: image is read, decoded, converted to tf.float32 data type; mask is created for a given image; if specified, a patch is cropped from an image and its mask; image and mask are resized to specific size if the size provided; image is normalized to [-1, 1] range (requirements of the chosen model for this task).
     
   - data augmentation:
In order to decrease the risk of overfitting and boost model's ability to geberalize data aumentation technique was used. All the augmentations were united under one custom class, which includes simple random horizontal/vertical flip (must be applied simultaneously to both image and its mask, therefore "seed" hyperameter was provided) and random brightness applied to the image.
     
6. Data Pipelines:
   Separate tf.data Datasets are set up for both training and validation:
    - train_ds: 1) generator yields batch of preprocessed training instances; 2) augmentations are applied using map() function; 3) dataset prefetching for performance; 4) repeating (for continuous data streaming during model training);
    - valid_ds: 1) generator yields batch of preprocessed training instances; 2) dataset prefetching;

5. Model
   - architecture:
Suggested model architecture for this task is U-Net, which is proven to work well with small to medium datasets. However, there can be different modifications to the original architecture, like different backbone models for image downsampling path, often with pretrained weights, which helps the model to converge faster and cuts training time. One of the general TensorFlow tutorials [2] was taken as a reference to create such a model. Here, the backbone is MobileNetV2 (loaded from keras.applications package) pretrained on imagenet dataset. Then matching custom image upsampling path is created and it is connected to a few intermediate layers from downsampling path to create skip connections. Model's high-level overview:
     ![model](https://github.com/OMarchevska/airbus_ship_detection_challenge/assets/84033554/b0aa7f3f-8afd-4d33-b540-1e796946c46e)

The whole model only has 6M parameters and it has inputs size requirements of [224, 224, 3] and also input values must be [-1, 1] range. The training dataset was prepared according to this requirements.

   - metrics:
     The training objective is the dice score, which measures the similarity between predicted and ground truth masks. It is defined as twice the intersection of the predicted and ground truth masks divided by the sum of their sizes.
       
   - loss:
   Loss is one the crutial parts of model training, since it greatly impacts learning process. Considering that the objective is the Dice Coefficient and there is a strong data imbalance we can't use conventional loss function for this task. The paper [3] described different loss functions depending on the use case. Specifically, the one that effectively deals with highly-imbalanced datasets is the Focal Loss, it works by down-weighting the contribution of easy examples, enabling model to learn hard examples. In addition to focal loss, top kagglers for this particular task also used Dice Loss. Dice Loss is inspired from Dice Coefficient, but due to its non-convex nature it has been modified to make it more tractable: 1 is added in numerator and denominator to ensure that the function is not undefined in edge case scenarios such as when y_true = y_pred = 0.
For this task the Mixed Loss is used, where the Dice Loss and Binary Focal Loss are combined and scaled to bring them to the same faces. Additionally we must take the logarithm of the Dice Loss which boosts the loss for the cases when the objects are not detected correctly and dice is close to zero. 


7. Training
   - baseline:
     Model was evaluared on the validation set prior to training:      (Dice Coefficient)
     
   - pretraining
     Since the model backbone was pretrained on the imagenet dataset, but the upsampling part of the model is not trained at all, we need to freeze trained weights in order to not to cause forgetting due to the large learning rate. First, the optimal learning rate needs to be established, which can be done using learning-rate/loss plot from the process of training model for just 1 epoch, dynamically changing model optimizer's learning rate and measuring model's losess obtained in the process. The code was borrowed  from Aurelien Geron github [4], but I slightly modified it to work with Keras 3 (my implementation code is located in the utils directory). Here is the generated plot :
     ![image](https://github.com/OMarchevska/airbus_ship_detection_challenge/assets/84033554/0cc19dcd-8c20-4c5d-a865-9d097f62ac2e)
The plot gives a hint on the maximum learning rate (training starts diverging), the optimal learning rate is typically chosen to be 10 times lower. The model was pretrained for only 10 epochs using. To facilitate the process several callbacks were incorporated:
   1. ReduceLROnPlateau callback (was configured to decrease learning rate by a factor of 0.2 whenever it sees no progress on the validation score for 1 epoch, with initial learning rate set to 1e-2 and maximum learning rate 1e-3);
   2. ModelCheckpoint callback (saves the best model based on the validation score during the training);
   3. EarlyStopping callback (interrupt training completely if there is no progress on validation score for the 3 epochs)
Optimizer - Adam is a default choice in most cases, in private runs was compared to a couple of other optimizers, which all performed almost identically good.

   - fine-tuning:
     Model weights (except for BatchNormalization layer weights) were unfrozen, new optimal learning rate indentified and the fine-tuning process was conducted using the same setup as for model pretraining for N number of epochs. The final fine-tuned model is stored in models directory in .keras format.
     Learning rate / loss plot for fine-tuning:
     ![image](https://github.com/OMarchevska/airbus_ship_detection_challenge/assets/84033554/d9b14f6e-d7ca-434d-a45b-f8d01b62a2fb)

9. Inference
   - data preparation:
     Separate functions are created to read unlabeled data for inference process. Given there are images to test in the data/test_v2, test.py file generates masks for them and saves in the results directory.
  









































References:
1. https://www.researchgate.net/publication/377933146_A_review_on_current_progress_of_semantic_segmentation
2. https://www.tensorflow.org/tutorials/images/segmentation
3. https://arxiv.org/pdf/2006.14822.pdf
4. https://github.com/ageron
