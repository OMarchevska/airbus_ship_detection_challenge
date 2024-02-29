This repository is dedicated to the segmentation model training for Airbus Ship Detection Challenge hosted by Kaggle:
- Task: Semantic Segmentation
- Data: https://www.kaggle.com/competitions/airbus-ship-detection/data


Required instrumets: requirements.txt

Repository structure overview:
1. 
2. 
3.

The process of tackling this problem consists of several substeps:

1. Problem domain investigation for:
   - existing solutions (could serve a purpose of a baseline model for further work and performance comparison);
   - latest research in the area (possible guidlines for optimal solutions space elimination) [1];


2. Data analysis:
   - data structure:
     labels: images of size 768x768;
     targets: image masks that are stored in numerical form in a file train_ship_segmentations_v2.csv; 
   - data overview:
The preliminary data overview is .. in the notebook showed strong data imbalance, where the number of images with no targets to segment (150000 (78.0% of total data)) is substantially higher than images that contain targets (42556 (22.0%)). The total number of data entries is 231723, where the number of unique images is 192556, which means that some images contain multiple objects to segment. Moreover, the targets themselves most of the time do not take up more than 5% of the total number of image. Problem domain investigation results suggested to discard all the "blank" images and continue with only "target-containing" ones. This allows significantly decrease training time and balances the training data.


3. Data Preprocessing:
   - training / validation splits generation:
     were created using sklearns train_test_split() function using 90/10 split for training and validation set correspondingly.
   - data transformation:
Due to different formats in which the input data stored separate preprocessing logic must be incorporated to load and process both images and label and then combine the pairs into training instances. To solve this task  
   - development of efficient data pipelines for training 
   - 

4. Model
   - architecture:
     U-Net architecture with MobileNetV2 (loaded from keras.applications package) backbone for image downsampling path was chosen, a few intermediate layers were picked as skips to connect downsampling and upsampling paths, the upsampling path itself was built using custom blocks with matching to skip connections shapes. Model's high-level overview:
     ![model](https://github.com/OMarchevska/airbus_ship_detection_challenge/assets/84033554/b0aa7f3f-8afd-4d33-b540-1e796946c46e)
     The whole model only has 6M parameters and has an option of loading pretrained on imagenet dataset weights, which in fact were used in the training process. However, model has inputs size requirements of [224, 224, 3] and also input values must be [-1, 1] range. 
       
   - loss:
   Loss is one the crutial parts of model training, since it greatly impacts learning process. Considering that the objective is the dice score, we need to force model improve this objective, which can be achieved using dice loss, which is just (1 - dice score). However, domain investigation results on this topic suggested consider mixed loss function one that can handle strong data imbalance 
   - metrics

5. Training
   - baseline
   - pretraining
   - fine-tuning
     
6. Inference
   
8. Error Analysis 
  









































References:
1. https://www.researchgate.net/publication/377933146_A_review_on_current_progress_of_semantic_segmentation
2. 
