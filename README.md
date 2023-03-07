# Fusion_model

A classification fusion model for Ultrasound images and DOT reconstruction images.
![This is an image](https://github.com/OpticalUltrasoundImaging/Fusion_model/blob/main/structures.png)

## Abstract

Diffuse optical tomography (DOT) is a promising technique that provides functional information related to tumor angiogenesis. However, reconstructing the DOT function map of a breast lesion is an ill-posed and underdetermined inverse process.  A co-registered ultrasound (US) system that provides structural information about the breast lesion can improve the localization and accuracy of DOT reconstruction. Additionally, the well-known US characteristics of benign and malignant breast lesions can further improve cancer diagnosis based on DOT alone. Inspired by a fusion model deep learning approach, we combined US features extracted by a modified VGG-11 network with images reconstructed from a DOT deep learning auto-encoder-based model to form a new neural network for breast cancer diagnosis. The combined neural network model was trained with simulation data and fine-tuned with clinical data: it achieved an AUC of 0.931 (95% CI: 0.919-0.943), superior to those achieved using US images alone (0.860) or DOT images alone (0.842).

## Details

Model structures can be found in net.py.
