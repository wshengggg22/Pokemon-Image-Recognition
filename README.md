**Pokémon Image Recognition (APS360 Project)**

A CNN-based deep learning system that identifies Pokémon from images across different visual styles — including sprites, anime artwork, and 3D renders.
Built for the APS360: Applied Fundamentals of Deep Learning course, this project demonstrates automated Pokémon recognition, with potential applications in digital Pokédex tools, educational games, and AI-driven content systems.

**Motivation**

The global popularity of Pokémon, with 1,025 distinct species as of 2025 according to the National Pokédex, provides an exciting opportunity to explore fine-grained image classification. Each Pokémon species features unique visual traits, making this a compelling deep learning problem.

Traditional image processing methods struggle with the wide variety of Pokémon styles (e.g., pixel sprites, anime, and 3D renders) because they depend on hand-crafted features that do not generalize well.
In contrast, Convolutional Neural Networks (CNNs) can automatically learn hierarchical visual representations — such as edges, textures, and color patterns — that remain consistent across these variations.

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
