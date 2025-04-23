Model Comparison on Stanford Cars & EuroSAT Datasets

Overview
This mini-project was carried out over a few hours as a quick comparative study of classical CNNs and modern transformer-based architectures on two distinct image classification tasks: fine-grained recognition (Stanford Cars) and satellite scene classification (EuroSAT). Despite the limited time, the analysis yielded several key insights about model behavior, convergence, and inductive biases.

Datasets

1. Stanford Cars
A fine-grained dataset for classifying 196 classes of cars, with a strong focus on visual similarities among models.

2. EuroSAT
A satellite scene classification dataset with 10 classes (e.g., forest, water, and urban) for land-cover classification.

Architectures Tested
1. VGG16 (with and without Batch Normalization)
2. ResNet18
3. SwinV2-Tiny
4. ViT-Base-16

Best Validation Scores
| S.No | Dataset  | Model used                | Best Validation | Step |
|------|----------|---------------------------|-----------------|------|
| 1    | Stanford | VIT-Base-16               | 0.53            | 5    |
| 2    | Stanford | VGG-11                    | 36.87           | 5    |
| 3    | Stanford | VGG-13                    | 38.1            | 26   |
| 4    | Stanford | VGG-16                    | 49.48           | 16   |
| 5    | Stanford | VGG-16 Lightweight        | 65.61           | 21   |
| 6    | Stanford | ResNet-18                 | 74.92           | 14   |
| 7    | Stanford | ResNet-34                 | 67.79           | 12   |
| 8    | Stanford | SwinV2-Tiny               | 78.27           | 30   |
| 9    | Stanford | VGG-19-BN Lightweight     | 78.38           | 12   |
| 10   | Stanford | VGG-16-BN Lightweight     | 79.38           | 20   |
| 11   | EuroSAT  | VIT-Base-16               | 100             | 5    |
| 12   | EuroSAT  | SwinV2-Tiny               | 100             | 5    |


Key Findings
1. The batch-normalized variant of VGG16 (VGG16_BN) slightly outperformed SwinV2-Tiny on the Stanford Cars dataset, especially when enhanced with Global Average Pooling (GAP) before classification.
2. The addition of GAP improved feature aggregation and reduced overfitting, while BatchNorm accelerated convergence and improved generalization.
3. SwinV2-Tiny performed competitively, with fast convergence and good generalization, but narrowly missed top accuracy.
4. ViT-Base-16 struggled with class distinctions in Stanford Cars due to the lack of spatial inductive bias and hierarchical structure.
5. On the EuroSAT dataset, both ViT and Swin achieved 100% test accuracy within 5 epochs, showcasing strong generalization on coarse-grained spatial data.
6. The batch-normalized VGG16 (VGG16_BN), when combined with GAP, demonstrated robust fine-grained classification performance, rivaling more modern transformer-based models.
7. Inductive biases—such as spatial locality in CNNs and hierarchical attention in Swin—were key to strong performance, especially in fine-grained visual tasks.

Other activities : GradCAM activation maps were used to interpret model predictions on the Stanford Cars dataset. 
