# Steel Defect Detection

This project implements semantic segmentation models to detect and classify steel surface defects into 4 different categories. A VGG-like Model and Lightweight EfficientNet U-Net model were trained on the Severstal Steel Defect Detection dataset.

## Exploratory Data Analysis
The data used for this project was downloaded from https://www.kaggle.com/competitions/severstal-steel-defect-detection/data. Each image may have no defects, a defect of a single class, or defects of multiple classes. 

A quick exploratory data analysis shows: 
* There are 4 defect classes
* The defect classes are imbalanced, with class 2 comprising only 3.4% of the data, while class 3 accounts for 72.5%; class 1 and 4 account for 12.6% and 11.3%, respectively. 

I implemented data augmentation on the training data for the minority classes, with augmentation_factors = [2, 6, 1, 2] for classes=[1, 2, 3, 4]. Horizontal/vertical flips, brightness adjustment are used to augment the dataset. 

## Model Construction
### Metrics for Evaluation
- **IoU (Intersection over Union)**: Primary segmentation metric, Best overall segmentation metric
- **Dice Coefficient**: Alternative segmentation metric, More sensitive to small defects than IoU

- **Recall**: Coverage of actual positives; critical for defect detection - "Did we catch all defects?". More important than precision for safety
- **Precision**: Accuracy of positive predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Per-class Accuracy**: Class-wise performance
### Model Architectures
1. **VGG-like Model**
- Simplified encoder-decoder without skip connections
- Lighter architecture for faster training
- Good baseline performance

2. **Lightweight EfficientNet U-Net** 
- EfficientNet-B0 as the encoder (pretrained on ImageNet)
- Custom U-Net decoder with skip connections
- Lightweight design for faster training
### Loss Function

Combined loss of *binary_crossentropy* and *dice loss* is used. *Dice Loss* is a segmentation loss function based on the Dice Coefficient (also known as F1 Score for binary classification). It's particularly effective for handling class imbalance in segmentation tasks like steel defect detection.

### Performance Optimization

Overfitting Mitigation Strategies:

- Early Stopping: Halt training if validation IoU doesn’t improve after 3 epochs.

- Learning Rate Scheduling: Halve the learning rate when validation performance plateaus.

- Dropout: Applied (rate 0.1–0.3) in decoder layers to reduce co-adaptation.

- Batch Normalization: Improves stability and generalization.

- Data Augmentation: Includes flips and brightness variations to expand training diversity.

## Model Evaluation

| Model | Val IoU | Val Dice |Val Recall | F1 Score | Training time |
|-------|---------|----------|-----------|----|---------------|
| VGG-like | 0.62 | 0.90| 0.61 | 0.73| faster |
| EfficientNet U-Net | 0.60 | 0.90 | 0.62 | 0.72 | slower|

# Summary 

Both the VGG-like model and the EfficientNet U-Net achieve similar Dice scores, indicating strong segmentation quality. However, the VGG-like model delivers slightly better IoU and recall, and trains much faster. 

While EfficientNet U-Net—with its pretrained encoder and skip connections—was expected to outperform, results show that the simpler VGG-like architecture is more effective for steel defect detection. This may be due to the domain gap between ImageNet pretraining and industrial steel surfaces, as well as the limited dataset size, which makes fine-tuning prone to overfitting. Overall, the VGG-like model’s simplicity and stability make it a better fit for this specialized task. Note that only a shallow VGG-like model was tested here; future work could explore deeper VGG variants with increased data augmentation to balance simplicity with higher representational capacity.




