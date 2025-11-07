**Pokémon Image Recognition (APS360 Project)**

A CNN-based deep learning system that identifies Pokémon from images across different visual styles — including sprites, anime artwork, and 3D renders.

Built for the APS360: Applied Fundamentals of Deep Learning course, this project demonstrates automated Pokémon recognition, with potential applications in digital Pokédex tools, educational games, and AI-driven content systems.

**Motivation**

The global popularity of Pokémon, with 1,025 distinct species as of 2025 according to the National Pokédex, provides an exciting opportunity to explore fine-grained image classification. Each Pokémon species features unique visual traits, making this a compelling deep learning problem. Traditional image processing methods struggle with the wide variety of Pokémon styles (e.g., pixel sprites, anime, and 3D renders) because they depend on hand-crafted features that do not generalize well. In contrast, Convolutional Neural Networks (CNNs) can automatically learn hierarchical visual representations — such as edges, textures, and color patterns — that remain consistent across these variations.

**Methodology**

This project fine-tunes a pretrained ResNet-18 model using transfer learning, adapting general visual knowledge from ImageNet to the Pokémon domain.

The model is trained on the 7,000 Labeled Pokémon Dataset from Kaggle, which contains images of 150 Pokémon species.

By leveraging transfer learning, the model gains flexibility to recognize Pokémon even when encountering unseen styles or renderings.

**Dataset Attribution**

Dataset: 7,000 Labeled Pokémon Dataset

Author: Lantian773030

License: For educational and research use only.

All Pokémon characters and images are © Nintendo / Game Freak / The Pokémon Company.

**Key Technologies**

PyTorch for model development

ResNet-18 (transfer learning)

Image preprocessing and augmentation

Cross-entropy loss & Adam optimizer

**Potential Applications**

Automated Pokédex tools

Educational games or quizzes

Visual Pokémon encyclopedia or classifier

**Acknowledgements**

This project was developed as part of APS360 – Applied Fundamentals of Deep Learning at the University of Toronto.

Special thanks to Lantian773030 for providing the Pokémon dataset used for model training.

**Dataset Preprocessing**

Dataset: [7,000 Labeled Pokémon Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/discussion?sort=hotness)

To reproduce the results, download the dataset and place it under a folder named pokemon_dataset/ in the project root. Then, run the split_dataset.py file, this will split the dataset into training, validation and testing dataset with a ratio of 70%, 15%, 15%, and the split dataset can be found in the directory "pokemon_split". 

**Data Loading and Transformation**

The data_loader.py file provides functions for loading and preprocessing the Pokémon dataset.

Training Data Augmentation: Applies transformations such as random rotation, horizontal flip, and color jitter to increase dataset diversity.

Data Loading: Functions are available to create PyTorch DataLoader objects for training, validation, and test sets.

Visualization: Includes helper functions to display sample images from the dataset, useful for verifying augmentations and preprocessing steps.

**Baseline Models**

I have trained two different baseline models: a logistic regression classifier and a k-Nearest Neighbors (k-NN) classifier. Both models were trained on flattened pixel values and color histograms extracted from resized Pokémon images (64×64). The logistic regression classifier achieved a validation accuracy of 30.79\%, while the k-NN classifier reached 14.76\%. These results provide a reasonable non-deep-learning baseline for evaluating the performance gains achieved with CNNs. 

To visualize the performance of the baseline models, I computed the confusion matrices for their predictions and plotted the 50 most misclassified classes for each model as heatmaps. In these heatmaps, the diagonal light spots represent correctly classified images, while the off-diagonal cells indicate misclassifications. The heatmaps show that both baseline models struggle to accurately classify Pokémon images, highlighting the need for a more robust classifier in the primary model.

![Logistic Regression classifier CM heatmap](/Logistic%20Regression%20classifier%20CM%20heatmap.png)
![KNN classifier CM heatmap](/KNN%20classifier%20CM%20heatmap.png)

**Primary Model**

For my primary classification model (refer to "pokemon_classifier.py"), I adopted transfer learning using the pre-trained ResNet-18 CNN. Input images were resized to 224 x 224 pixels, as recommended for ResNet-18. I replaced the original final layer with a custom two-layer classifier (512 → 256 → 150) featuring ReLU activation and a 0.5 dropout for regularization. All pre-trained feature extractor weights were frozen, so only the new head was trained using Cross-Entropy Loss and the Adam optimizer with weight decay over 30 epochs.