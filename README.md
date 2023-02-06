# EVA8_Session6_AdvancedConcepts

### Problem Statement
Train a CNN classifier model to handle CIFAR-10 Dataset.


#### MUST!!

1. change the code such that it uses GPU and
2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
3. Total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. use albumentation library and apply:
        1. ```horizontal flip```
        2. ```shiftScaleRotate```
        3. ```coarseDropout (max_holes = 1, max_height=16px, max_width=16px, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)```
8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

### Solution
#### Please refer to the [Assignment_s6_Solution.ipynb](/Assignment_s6_Solution.ipynb) Jupyter Notebook for the experiments, helper code and plots.

#### Modules - API
#### [model.py](/model.py)
1. This is the main module that contains the code for the Model definition.
2. The module contains 2 models(3 classes):
    1. DilatedConv class: This class implements the version of ```maxpooling2D```, and creates the same RF effect that MaxPooling layer would do.
    2. CIFAR10CNN2: This model(class) contains CNN layers, MaxPool2D, Dilated and depthwise sepearble covolution layers.
    3. CIFAR10CNN3: This model(class) contains the CNN layers, DilatedConv (substitute for MaxPool Layer), dilated and depth wise convolution layers.

#### [utils.py](/utils.py)
1. This is a helper file that contains the source code(class) ```Cifar10Dataset``` to generate the CIFAR-10 dataset.   

#### [train.py](/train.py)
1. This is a helper module to load the requried model from ```model.py``` and performs the following actions:
    ```train_model```: Trains the model (using trainloader)
    ```test_model```: Tests the model (testloader)
    ```get_stats()```: method that tracks the ```loss``` and ```accuracy``` metrics (both train and validation data)

### CIFAR10CNN2 (Uses Conv2d, MaxPool2d, dilation and depth-wise conv layers):
##### Model Summary
```
Device: cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 30, 30]             896
       BatchNorm2d-2           [-1, 32, 30, 30]              64
              ReLU-3           [-1, 32, 30, 30]               0
            Conv2d-4           [-1, 64, 28, 28]             640
            Conv2d-5           [-1, 64, 28, 28]           4,160
       BatchNorm2d-6           [-1, 64, 28, 28]             128
              ReLU-7           [-1, 64, 28, 28]               0
            Conv2d-8           [-1, 64, 28, 28]             640
            Conv2d-9           [-1, 64, 28, 28]           4,160
      BatchNorm2d-10           [-1, 64, 28, 28]             128
             ReLU-11           [-1, 64, 28, 28]               0
           Conv2d-12           [-1, 64, 28, 28]             640
           Conv2d-13           [-1, 64, 28, 28]           4,160
      BatchNorm2d-14           [-1, 64, 28, 28]             128
             ReLU-15           [-1, 64, 28, 28]               0
           Conv2d-16           [-1, 64, 26, 26]          36,928
      BatchNorm2d-17           [-1, 64, 26, 26]             128
             ReLU-18           [-1, 64, 26, 26]               0
        MaxPool2d-19           [-1, 64, 13, 13]               0
           Conv2d-20           [-1, 64, 11, 11]             640
           Conv2d-21           [-1, 64, 11, 11]           4,160
      BatchNorm2d-22           [-1, 64, 11, 11]             128
             ReLU-23           [-1, 64, 11, 11]               0
           Conv2d-24           [-1, 64, 11, 11]             640
           Conv2d-25           [-1, 64, 11, 11]           4,160
      BatchNorm2d-26           [-1, 64, 11, 11]             128
             ReLU-27           [-1, 64, 11, 11]               0
           Conv2d-28            [-1, 128, 7, 7]          73,856
      BatchNorm2d-29            [-1, 128, 7, 7]             256
             ReLU-30            [-1, 128, 7, 7]               0
           Conv2d-31            [-1, 256, 5, 5]           2,560
           Conv2d-32             [-1, 64, 5, 5]          16,448
      BatchNorm2d-33             [-1, 64, 5, 5]             128
        AvgPool2d-34             [-1, 64, 1, 1]               0
           Linear-35                   [-1, 10]             650
================================================================
Total params: 156,554
Trainable params: 156,554
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.02
Params size (MB): 0.60
Estimated Total Size (MB): 7.62
----------------------------------------------------------------
```

###### Training Stats:

```
EPOCH: 0
Loss=1.319269061088562 Batch_id=390 Accuracy=42.49: 100%|████████████████████████████| 391/391 [00:20<00:00, 18.63it/s]

Test set: Average loss: 0.0096, Accuracy: 5682/10000 (56.82%)

EPOCH: 1
Loss=1.19991934299469 Batch_id=390 Accuracy=53.90: 100%|█████████████████████████████| 391/391 [00:21<00:00, 18.20it/s]

Test set: Average loss: 0.0080, Accuracy: 6323/10000 (63.23%)

EPOCH: 2
Loss=1.1187125444412231 Batch_id=390 Accuracy=58.46: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.0075, Accuracy: 6696/10000 (66.96%)

EPOCH: 3
Loss=1.1585661172866821 Batch_id=390 Accuracy=61.76: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.31it/s]

Test set: Average loss: 0.0066, Accuracy: 7093/10000 (70.93%)

EPOCH: 4
Loss=0.8349015116691589 Batch_id=390 Accuracy=64.04: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.28it/s]

Test set: Average loss: 0.0062, Accuracy: 7227/10000 (72.27%)

EPOCH: 5
Loss=1.217560052871704 Batch_id=390 Accuracy=65.58: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.00it/s]

Test set: Average loss: 0.0068, Accuracy: 7063/10000 (70.63%)

EPOCH: 6
Loss=1.1786081790924072 Batch_id=390 Accuracy=66.87: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.31it/s]

Test set: Average loss: 0.0057, Accuracy: 7513/10000 (75.13%)

EPOCH: 7
Loss=0.7552996873855591 Batch_id=390 Accuracy=67.68: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.28it/s]

Test set: Average loss: 0.0054, Accuracy: 7636/10000 (76.36%)

EPOCH: 8
Loss=0.9880145192146301 Batch_id=390 Accuracy=68.87: 100%|███████████████████████████| 391/391 [00:21<00:00, 17.84it/s]

Test set: Average loss: 0.0052, Accuracy: 7681/10000 (76.81%)

EPOCH: 9
Loss=0.794843316078186 Batch_id=390 Accuracy=69.78: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.22it/s]

Test set: Average loss: 0.0049, Accuracy: 7830/10000 (78.30%)

EPOCH: 10
Loss=1.0437480211257935 Batch_id=390 Accuracy=70.40: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.0049, Accuracy: 7830/10000 (78.30%)

EPOCH: 11
Loss=0.7529229521751404 Batch_id=390 Accuracy=70.96: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.20it/s]

Test set: Average loss: 0.0051, Accuracy: 7793/10000 (77.93%)

EPOCH: 12
Loss=0.7159638404846191 Batch_id=390 Accuracy=71.56: 100%|███████████████████████████| 391/391 [00:23<00:00, 16.29it/s]

Test set: Average loss: 0.0047, Accuracy: 7995/10000 (79.95%)

EPOCH: 13
Loss=0.7878557443618774 Batch_id=390 Accuracy=72.06: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.52it/s]

Test set: Average loss: 0.0046, Accuracy: 7968/10000 (79.68%)

EPOCH: 14
Loss=0.6853529214859009 Batch_id=390 Accuracy=72.28: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.29it/s]

Test set: Average loss: 0.0044, Accuracy: 8046/10000 (80.46%)

EPOCH: 15
Loss=0.5699108839035034 Batch_id=390 Accuracy=73.37: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.24it/s]

Test set: Average loss: 0.0046, Accuracy: 8007/10000 (80.07%)

EPOCH: 16
Loss=0.859006404876709 Batch_id=390 Accuracy=73.25: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.20it/s]

Test set: Average loss: 0.0045, Accuracy: 8056/10000 (80.56%)

EPOCH: 17
Loss=0.6570026278495789 Batch_id=390 Accuracy=73.55: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.62it/s]

Test set: Average loss: 0.0045, Accuracy: 7984/10000 (79.84%)

EPOCH: 18
Loss=0.739359438419342 Batch_id=390 Accuracy=74.28: 100%|████████████████████████████| 391/391 [00:20<00:00, 18.82it/s]

Test set: Average loss: 0.0042, Accuracy: 8150/10000 (81.50%)

EPOCH: 19
Loss=0.8534054756164551 Batch_id=390 Accuracy=74.44: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.87it/s]

Test set: Average loss: 0.0044, Accuracy: 8096/10000 (80.96%)

EPOCH: 20
Loss=0.8753746747970581 Batch_id=390 Accuracy=74.90: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.85it/s]

Test set: Average loss: 0.0042, Accuracy: 8192/10000 (81.92%)

EPOCH: 21
Loss=0.6157063245773315 Batch_id=390 Accuracy=75.07: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.69it/s]

Test set: Average loss: 0.0040, Accuracy: 8263/10000 (82.63%)

EPOCH: 22
Loss=0.7166267037391663 Batch_id=390 Accuracy=75.62: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.74it/s]

Test set: Average loss: 0.0042, Accuracy: 8213/10000 (82.13%)

EPOCH: 23
Loss=0.6345826387405396 Batch_id=390 Accuracy=75.78: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.82it/s]

Test set: Average loss: 0.0039, Accuracy: 8313/10000 (83.13%)

EPOCH: 24
Loss=0.7625342011451721 Batch_id=390 Accuracy=76.12: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.83it/s]

Test set: Average loss: 0.0042, Accuracy: 8222/10000 (82.22%)

EPOCH: 25
Loss=0.5149210095405579 Batch_id=390 Accuracy=76.04: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.31it/s]

Test set: Average loss: 0.0042, Accuracy: 8166/10000 (81.66%)

EPOCH: 26
Loss=0.46139341592788696 Batch_id=390 Accuracy=76.29: 100%|██████████████████████████| 391/391 [00:21<00:00, 18.20it/s]

Test set: Average loss: 0.0046, Accuracy: 8108/10000 (81.08%)

EPOCH: 27
Loss=0.7815054655075073 Batch_id=390 Accuracy=76.48: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.53it/s]

Test set: Average loss: 0.0039, Accuracy: 8320/10000 (83.20%)

EPOCH: 28
Loss=0.776668906211853 Batch_id=390 Accuracy=77.08: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.0039, Accuracy: 8311/10000 (83.11%)

EPOCH: 29
Loss=0.4954877495765686 Batch_id=390 Accuracy=77.00: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.03it/s]

Test set: Average loss: 0.0046, Accuracy: 8061/10000 (80.61%)

EPOCH: 30
Loss=0.8496403694152832 Batch_id=390 Accuracy=77.23: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.37it/s]

Test set: Average loss: 0.0038, Accuracy: 8331/10000 (83.31%)

EPOCH: 31
Loss=0.630569338798523 Batch_id=390 Accuracy=77.17: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.44it/s]

Test set: Average loss: 0.0039, Accuracy: 8300/10000 (83.00%)

EPOCH: 32
Loss=0.6705737709999084 Batch_id=390 Accuracy=77.54: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.65it/s]

Test set: Average loss: 0.0037, Accuracy: 8410/10000 (84.10%)

EPOCH: 33
Loss=0.7090975046157837 Batch_id=390 Accuracy=78.00: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.48it/s]

Test set: Average loss: 0.0038, Accuracy: 8386/10000 (83.86%)

EPOCH: 34
Loss=0.6069981455802917 Batch_id=390 Accuracy=77.84: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.48it/s]

Test set: Average loss: 0.0038, Accuracy: 8369/10000 (83.69%)

EPOCH: 35
Loss=0.42124900221824646 Batch_id=390 Accuracy=77.93: 100%|██████████████████████████| 391/391 [00:21<00:00, 18.32it/s]

Test set: Average loss: 0.0041, Accuracy: 8257/10000 (82.57%)

EPOCH: 36
Loss=0.6059666872024536 Batch_id=390 Accuracy=78.17: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.40it/s]

Test set: Average loss: 0.0038, Accuracy: 8400/10000 (84.00%)

EPOCH: 37
Loss=0.6996375322341919 Batch_id=390 Accuracy=78.54: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.32it/s]

Test set: Average loss: 0.0035, Accuracy: 8488/10000 (84.88%)

EPOCH: 38
Loss=0.768610954284668 Batch_id=390 Accuracy=78.33: 100%|████████████████████████████| 391/391 [00:20<00:00, 18.77it/s]

Test set: Average loss: 0.0035, Accuracy: 8468/10000 (84.68%)

EPOCH: 39
Loss=0.8966341018676758 Batch_id=390 Accuracy=78.57: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.64it/s]

Test set: Average loss: 0.0039, Accuracy: 8365/10000 (83.65%)

EPOCH: 40
Loss=0.6685110926628113 Batch_id=390 Accuracy=78.57: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.68it/s]

Test set: Average loss: 0.0035, Accuracy: 8487/10000 (84.87%)

EPOCH: 41
Loss=0.5504958629608154 Batch_id=390 Accuracy=78.92: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.78it/s]

Test set: Average loss: 0.0036, Accuracy: 8497/10000 (84.97%)

EPOCH: 42
Loss=0.5209516882896423 Batch_id=390 Accuracy=79.31: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.80it/s]

Test set: Average loss: 0.0035, Accuracy: 8494/10000 (84.94%)

EPOCH: 43
Loss=0.7927006483078003 Batch_id=390 Accuracy=79.05: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.65it/s]

Test set: Average loss: 0.0038, Accuracy: 8419/10000 (84.19%)

EPOCH: 44
Loss=0.6614179611206055 Batch_id=390 Accuracy=79.23: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.94it/s]

Test set: Average loss: 0.0035, Accuracy: 8536/10000 (85.36%)

EPOCH: 45
Loss=0.5826968550682068 Batch_id=390 Accuracy=79.26: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.73it/s]

Test set: Average loss: 0.0033, Accuracy: 8589/10000 (85.89%)

EPOCH: 46
Loss=0.5718923807144165 Batch_id=390 Accuracy=79.53: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.69it/s]

Test set: Average loss: 0.0035, Accuracy: 8574/10000 (85.74%)

EPOCH: 47
Loss=0.5770961046218872 Batch_id=390 Accuracy=79.80: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.73it/s]

Test set: Average loss: 0.0036, Accuracy: 8475/10000 (84.75%)

EPOCH: 48
Loss=0.652831494808197 Batch_id=390 Accuracy=79.93: 100%|████████████████████████████| 391/391 [00:20<00:00, 18.78it/s]

Test set: Average loss: 0.0035, Accuracy: 8553/10000 (85.53%)

EPOCH: 49
Loss=0.4675709307193756 Batch_id=390 Accuracy=79.70: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.73it/s]

Test set: Average loss: 0.0037, Accuracy: 8466/10000 (84.66%)

EPOCH: 50
Loss=0.6425091624259949 Batch_id=390 Accuracy=79.92: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.79it/s]

Test set: Average loss: 0.0035, Accuracy: 8526/10000 (85.26%)

EPOCH: 51
Loss=0.5289527177810669 Batch_id=390 Accuracy=80.41: 100%|███████████████████████████| 391/391 [00:20<00:00, 18.68it/s]

Test set: Average loss: 0.0035, Accuracy: 8552/10000 (85.52%)

EPOCH: 52
Loss=0.4432052671909332 Batch_id=390 Accuracy=80.00: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.28it/s]

Test set: Average loss: 0.0034, Accuracy: 8568/10000 (85.68%)

EPOCH: 53
Loss=0.5429993271827698 Batch_id=390 Accuracy=80.13: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.29it/s]

Test set: Average loss: 0.0034, Accuracy: 8603/10000 (86.03%)

EPOCH: 54
Loss=0.459725558757782 Batch_id=390 Accuracy=80.37: 100%|████████████████████████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 0.0034, Accuracy: 8598/10000 (85.98%)

EPOCH: 55
Loss=0.6729629635810852 Batch_id=390 Accuracy=80.73: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.42it/s]

Test set: Average loss: 0.0035, Accuracy: 8549/10000 (85.49%)

EPOCH: 56
Loss=0.5055030584335327 Batch_id=390 Accuracy=80.27: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 0.0033, Accuracy: 8579/10000 (85.79%)

EPOCH: 57
Loss=0.6432158946990967 Batch_id=390 Accuracy=80.68: 100%|███████████████████████████| 391/391 [00:21<00:00, 18.41it/s]

Test set: Average loss: 0.0035, Accuracy: 8510/10000 (85.10%)

```

#### CIFAR10CNN3 (Uses Conv2d, dilated(instead of MaxPool2d) and depth-wise conv layers)
##### Model Summary
```
Device: cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 30, 30]             896
       BatchNorm2d-2           [-1, 32, 30, 30]              64
              ReLU-3           [-1, 32, 30, 30]               0
            Conv2d-4           [-1, 64, 28, 28]             640
            Conv2d-5           [-1, 64, 28, 28]           4,160
       BatchNorm2d-6           [-1, 64, 28, 28]             128
              ReLU-7           [-1, 64, 28, 28]               0
            Conv2d-8           [-1, 64, 28, 28]             640
            Conv2d-9           [-1, 64, 28, 28]           4,160
      BatchNorm2d-10           [-1, 64, 28, 28]             128
             ReLU-11           [-1, 64, 28, 28]               0
           Conv2d-12           [-1, 64, 28, 28]             640
           Conv2d-13           [-1, 64, 28, 28]           4,160
      BatchNorm2d-14           [-1, 64, 28, 28]             128
             ReLU-15           [-1, 64, 28, 28]               0
           Conv2d-16           [-1, 64, 28, 28]             640
           Conv2d-17           [-1, 64, 28, 28]           4,160
      BatchNorm2d-18           [-1, 64, 28, 28]             128
             ReLU-19           [-1, 64, 28, 28]               0
           Conv2d-20           [-1, 64, 28, 28]          36,928
      DilatedConv-21           [-1, 64, 28, 28]               0
           Conv2d-22           [-1, 64, 28, 28]             640
           Conv2d-23           [-1, 64, 28, 28]           4,160
      BatchNorm2d-24           [-1, 64, 28, 28]             128
             ReLU-25           [-1, 64, 28, 28]               0
           Conv2d-26          [-1, 128, 28, 28]           1,280
           Conv2d-27           [-1, 64, 28, 28]           8,256
      BatchNorm2d-28           [-1, 64, 28, 28]             128
             ReLU-29           [-1, 64, 28, 28]               0
           Conv2d-30          [-1, 128, 28, 28]           1,280
           Conv2d-31           [-1, 64, 28, 28]           8,256
      BatchNorm2d-32           [-1, 64, 28, 28]             128
             ReLU-33           [-1, 64, 28, 28]               0
           Conv2d-34           [-1, 64, 28, 28]             640
           Conv2d-35           [-1, 64, 28, 28]           4,160
      BatchNorm2d-36           [-1, 64, 28, 28]             128
             ReLU-37           [-1, 64, 28, 28]               0
           Conv2d-38           [-1, 64, 28, 28]             640
           Conv2d-39           [-1, 64, 28, 28]           4,160
      BatchNorm2d-40           [-1, 64, 28, 28]             128
             ReLU-41           [-1, 64, 28, 28]               0
           Conv2d-42          [-1, 128, 24, 24]          73,856
      BatchNorm2d-43          [-1, 128, 24, 24]             256
             ReLU-44          [-1, 128, 24, 24]               0
           Conv2d-45          [-1, 256, 22, 22]           2,560
           Conv2d-46           [-1, 64, 22, 22]          16,448
      BatchNorm2d-47           [-1, 64, 22, 22]             128
        AvgPool2d-48             [-1, 64, 1, 1]               0
           Linear-49                   [-1, 10]             650
================================================================
Total params: 185,610
Trainable params: 185,610
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 19.08
Params size (MB): 0.71
Estimated Total Size (MB): 19.80
----------------------------------------------------------------
```

##### Model training stats:
```

EPOCH: 0
Loss=1.5014398097991943 Batch_id=390 Accuracy=40.91: 100%|███████████████████████████| 391/391 [00:35<00:00, 11.14it/s]

Test set: Average loss: 0.0099, Accuracy: 5368/10000 (53.68%)

EPOCH: 1
Loss=1.1479570865631104 Batch_id=390 Accuracy=52.95: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.20it/s]

Test set: Average loss: 0.0136, Accuracy: 4559/10000 (45.59%)

EPOCH: 2
Loss=0.9432692527770996 Batch_id=390 Accuracy=57.72: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.27it/s]

Test set: Average loss: 0.0075, Accuracy: 6556/10000 (65.56%)

EPOCH: 3
Loss=1.0358648300170898 Batch_id=390 Accuracy=61.46: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.24it/s]

Test set: Average loss: 0.0071, Accuracy: 6852/10000 (68.52%)

EPOCH: 4
Loss=1.256654977798462 Batch_id=390 Accuracy=63.93: 100%|████████████████████████████| 391/391 [00:35<00:00, 11.06it/s]

Test set: Average loss: 0.0078, Accuracy: 6531/10000 (65.31%)

EPOCH: 5
Loss=0.8575426340103149 Batch_id=390 Accuracy=65.52: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.27it/s]

Test set: Average loss: 0.0062, Accuracy: 7258/10000 (72.58%)

EPOCH: 6
Loss=1.0261075496673584 Batch_id=390 Accuracy=66.97: 100%|███████████████████████████| 391/391 [00:35<00:00, 11.11it/s]

Test set: Average loss: 0.0064, Accuracy: 7183/10000 (71.83%)

EPOCH: 7
Loss=0.8462732434272766 Batch_id=390 Accuracy=68.23: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.21it/s]

Test set: Average loss: 0.0058, Accuracy: 7406/10000 (74.06%)

EPOCH: 8
Loss=0.7818406820297241 Batch_id=390 Accuracy=69.21: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.31it/s]

Test set: Average loss: 0.0057, Accuracy: 7543/10000 (75.43%)

EPOCH: 9
Loss=0.9040431976318359 Batch_id=390 Accuracy=70.39: 100%|███████████████████████████| 391/391 [00:35<00:00, 11.07it/s]

Test set: Average loss: 0.0053, Accuracy: 7681/10000 (76.81%)

EPOCH: 10
Loss=0.6373404264450073 Batch_id=390 Accuracy=71.26: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.24it/s]

Test set: Average loss: 0.0062, Accuracy: 7410/10000 (74.10%)

EPOCH: 11
Loss=0.7494758367538452 Batch_id=390 Accuracy=71.82: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.27it/s]

Test set: Average loss: 0.0050, Accuracy: 7876/10000 (78.76%)

EPOCH: 12
Loss=0.896914005279541 Batch_id=390 Accuracy=72.43: 100%|████████████████████████████| 391/391 [00:34<00:00, 11.21it/s]

Test set: Average loss: 0.0052, Accuracy: 7742/10000 (77.42%)

EPOCH: 13
Loss=0.7839080691337585 Batch_id=390 Accuracy=72.88: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.20it/s]

Test set: Average loss: 0.0052, Accuracy: 7799/10000 (77.99%)

EPOCH: 14
Loss=0.7451421022415161 Batch_id=390 Accuracy=73.20: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.20it/s]

Test set: Average loss: 0.0051, Accuracy: 7831/10000 (78.31%)

EPOCH: 15
Loss=0.9805407524108887 Batch_id=390 Accuracy=73.90: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.30it/s]

Test set: Average loss: 0.0048, Accuracy: 7950/10000 (79.50%)

EPOCH: 16
Loss=0.6375554800033569 Batch_id=390 Accuracy=74.86: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.24it/s]

Test set: Average loss: 0.0046, Accuracy: 8004/10000 (80.04%)

EPOCH: 17
Loss=0.9438649415969849 Batch_id=390 Accuracy=74.95: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.29it/s]

Test set: Average loss: 0.0046, Accuracy: 7940/10000 (79.40%)

EPOCH: 18
Loss=0.7998576760292053 Batch_id=390 Accuracy=75.22: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.27it/s]

Test set: Average loss: 0.0050, Accuracy: 8003/10000 (80.03%)

EPOCH: 19
Loss=0.744343101978302 Batch_id=390 Accuracy=75.65: 100%|████████████████████████████| 391/391 [00:34<00:00, 11.23it/s]

Test set: Average loss: 0.0043, Accuracy: 8180/10000 (81.80%)

EPOCH: 20
Loss=0.895769476890564 Batch_id=390 Accuracy=75.85: 100%|████████████████████████████| 391/391 [00:34<00:00, 11.23it/s]

Test set: Average loss: 0.0045, Accuracy: 8089/10000 (80.89%)

EPOCH: 21
Loss=0.7910277247428894 Batch_id=390 Accuracy=76.52: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.34it/s]

Test set: Average loss: 0.0045, Accuracy: 8079/10000 (80.79%)

EPOCH: 22
Loss=0.5713221430778503 Batch_id=390 Accuracy=76.47: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.25it/s]

Test set: Average loss: 0.0043, Accuracy: 8110/10000 (81.10%)

EPOCH: 23
Loss=0.6411437392234802 Batch_id=390 Accuracy=77.02: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.25it/s]

Test set: Average loss: 0.0053, Accuracy: 7706/10000 (77.06%)

EPOCH: 24
Loss=0.5955763459205627 Batch_id=390 Accuracy=77.14: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.31it/s]

Test set: Average loss: 0.0038, Accuracy: 8336/10000 (83.36%)

EPOCH: 25
Loss=0.6910505294799805 Batch_id=390 Accuracy=77.39: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.24it/s]

Test set: Average loss: 0.0041, Accuracy: 8284/10000 (82.84%)

EPOCH: 26
Loss=0.7630403637886047 Batch_id=390 Accuracy=77.81: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.26it/s]

Test set: Average loss: 0.0040, Accuracy: 8286/10000 (82.86%)

EPOCH: 27
Loss=0.5204685926437378 Batch_id=390 Accuracy=77.80: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.20it/s]

Test set: Average loss: 0.0039, Accuracy: 8311/10000 (83.11%)

EPOCH: 28
Loss=0.6721581816673279 Batch_id=390 Accuracy=78.27: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.40it/s]

Test set: Average loss: 0.0042, Accuracy: 8292/10000 (82.92%)

EPOCH: 29
Loss=0.7748070955276489 Batch_id=390 Accuracy=78.55: 100%|███████████████████████████| 391/391 [00:35<00:00, 11.09it/s]

Test set: Average loss: 0.0037, Accuracy: 8411/10000 (84.11%)

EPOCH: 30
Loss=0.4432109296321869 Batch_id=390 Accuracy=78.43: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.31it/s]

Test set: Average loss: 0.0042, Accuracy: 8189/10000 (81.89%)

EPOCH: 31
Loss=0.5772513151168823 Batch_id=390 Accuracy=78.51: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.44it/s]

Test set: Average loss: 0.0041, Accuracy: 8286/10000 (82.86%)

EPOCH: 32
Loss=0.7131236791610718 Batch_id=390 Accuracy=78.96: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.28it/s]

Test set: Average loss: 0.0034, Accuracy: 8504/10000 (85.04%)

EPOCH: 33
Loss=0.6308951377868652 Batch_id=390 Accuracy=79.39: 100%|███████████████████████████| 391/391 [00:35<00:00, 10.89it/s]

Test set: Average loss: 0.0037, Accuracy: 8424/10000 (84.24%)

EPOCH: 34
Loss=0.7656244039535522 Batch_id=390 Accuracy=79.45: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.28it/s]

Test set: Average loss: 0.0046, Accuracy: 8175/10000 (81.75%)

EPOCH: 35
Loss=0.7443010807037354 Batch_id=390 Accuracy=79.67: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.29it/s]

Test set: Average loss: 0.0036, Accuracy: 8494/10000 (84.94%)

EPOCH: 36
Loss=0.48778969049453735 Batch_id=390 Accuracy=79.73: 100%|██████████████████████████| 391/391 [00:34<00:00, 11.34it/s]

Test set: Average loss: 0.0035, Accuracy: 8476/10000 (84.76%)

EPOCH: 37
Loss=0.7194676995277405 Batch_id=390 Accuracy=79.96: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.34it/s]

Test set: Average loss: 0.0041, Accuracy: 8310/10000 (83.10%)

EPOCH: 38
Loss=0.6228103637695312 Batch_id=390 Accuracy=80.15: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.37it/s]

Test set: Average loss: 0.0036, Accuracy: 8429/10000 (84.29%)

EPOCH: 39
Loss=0.4868471026420593 Batch_id=390 Accuracy=79.99: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.35it/s]

Test set: Average loss: 0.0037, Accuracy: 8425/10000 (84.25%)

EPOCH: 40
Loss=0.6079707145690918 Batch_id=390 Accuracy=80.23: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.35it/s]

Test set: Average loss: 0.0041, Accuracy: 8313/10000 (83.13%)

EPOCH: 41
Loss=0.6803017854690552 Batch_id=390 Accuracy=80.66: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.34it/s]

Test set: Average loss: 0.0033, Accuracy: 8604/10000 (86.04%)

EPOCH: 42
Loss=0.8316338658332825 Batch_id=390 Accuracy=80.52: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.37it/s]

Test set: Average loss: 0.0035, Accuracy: 8488/10000 (84.88%)

EPOCH: 43
Loss=0.5814430713653564 Batch_id=390 Accuracy=80.97: 100%|███████████████████████████| 391/391 [00:35<00:00, 11.05it/s]

Test set: Average loss: 0.0036, Accuracy: 8498/10000 (84.98%)

EPOCH: 44
Loss=0.7128027677536011 Batch_id=390 Accuracy=81.10: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.26it/s]

Test set: Average loss: 0.0036, Accuracy: 8493/10000 (84.93%)

EPOCH: 45
Loss=0.4303329885005951 Batch_id=390 Accuracy=81.05: 100%|███████████████████████████| 391/391 [00:34<00:00, 11.28it/s]

Test set: Average loss: 0.0039, Accuracy: 8393/10000 (83.93%)

EPOCH: 46
Loss=0.47404804825782776 Batch_id=390 Accuracy=81.04: 100%|██████████████████████████| 391/391 [00:34<00:00, 11.30it/s]

Test set: Average loss: 0.0038, Accuracy: 8461/10000 (84.61%)

EPOCH: 47
Loss=0.41682443022727966 Batch_id=390 Accuracy=81.12: 100%|██████████████████████████| 391/391 [00:34<00:00, 11.18it/s]

Test set: Average loss: 0.0033, Accuracy: 8584/10000 (85.84%)
```