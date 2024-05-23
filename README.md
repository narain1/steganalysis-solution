# Project Overview

This project focused on detecting hidden information in steganographic images using advanced convolutional neural network (CNN) techniques. Key components included the implementation of lower stride CNNs, application of modern image training techniques, and the creation of a synthetic dataset using various steganographic algorithms.
Techniques and Methods
Lower Stride Convolutional Neural Networks (CNNs)

* Objective: Improve the granularity of feature detection in images.
* Implementation: Utilized lower stride CNNs to capture finer details essential for identifying hidden information in steganographic images.

## Modern Image Training Techniques

* Mixup: Combined two images to create a new training sample, enhancing model generalization.
* CutMix: Mixed parts of two images, providing more robust training data.
* Label Smoothing: Applied regularization to prevent overfitting by reducing model confidence in predictions.

## Color Space Domains

* RGB: Used the standard color space common in most image processing tasks.
* YCbCr: Implemented to leverage luminance and chrominance separation, aiding in steganographic content detection.

Synthetic Dataset Creation
Steganographic Algorithms

* JMiPOD: Designed for minimal distortion steganography.
* JUNIWARD: Utilizes a universal wavelet relative distortion model for effective data embedding.
* UERD: Employs an edge-adaptive model to embed information, minimizing visual distortions.

Dataset Generation

* Process: Applied the above algorithms to various images to create a synthetic dataset.
* Purpose: Used the dataset to train and evaluate CNN models, ensuring comprehensive understanding of different steganographic techniques.

Results and Contributions

* Successfully implemented lower stride CNNs with modern training techniques, achieving significant improvements in detecting hidden information.
* Demonstrated the effectiveness of using both RGB and YCbCr color spaces for steganographic detection.
* Developed a robust synthetic dataset for further research and development in steganography detection.

Impact

This project advanced steganographic detection techniques, providing valuable insights and tools for future research in this field.
