# -*- coding: utf-8 -*-
# Written by Menghao Zhang on Fusion classification model paper
# List of python package name of version
# Python version: 3.7
# Package name and version: torch(pytorch): 1.10.0; torchvision: 0.11.1;
# PIL(pillow): 8.2.0, numpy: 1.21.2, scipy: 1.7.1
# Input variable type and dimension: US images: W*H*3; DOT images: single wavelength reconstructed images: 33*33*7
# n_depth: one float number, depth: one float number, z_radius: one float number
# Output: Fusion model probability: one float number

import torch
import nets
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.io as sio

# Define Identity class used in US feature extraction
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
# US feature extraction model, modified from VGG-11 network
def us_feature_extract(inputs):
    # Define device used in the model (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load initial VGG-11 model
    vgg11 = models.vgg11_bn(pretrained=True)
    # Get feature extraction layers
    idx_layer = 0
    for param in vgg11.features.parameters():
        idx_layer = idx_layer + 1
        if idx_layer > 9:
            break
        param.requires_grad = False
    # Image preprocessing based on the ImageNet neural network
    preprocess_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Newly created modules have require_grad=True by default
    # Use PIL package to generate input feed into the neural network
    PIL_image = Image.fromarray(np.uint8(inputs)).convert('RGB')
    input_data = preprocess_resize(PIL_image)
    input_datas = torch.unsqueeze(input_data, dim=0)
    # Modified VGG-11 network structure
    feature_features = list(vgg11.features.children())[:-4]
    # feature_features[18:21]=[]
    feature_features.extend([nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                             nn.ZeroPad2d((1,0,1,0)), 
                             nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                             nn.ReLU(inplace=True)])
    vgg11.features = nn.Sequential(*feature_features)
    num_features = vgg11.classifier[6].in_features
    classfy_features = list(vgg11.classifier.children())[3:-1]
    classfy_features.extend([nn.Linear(num_features, 512), nn.ReLU(),
                             nn.Dropout(p=0.5, inplace=False),
                             nn.Linear(512, 2)
                             ]) 
    vgg11.classifier = nn.Sequential(*classfy_features) # Replace the model classifier
    vgg11.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,8))
    # Load the trained weight
    vgg11.load_state_dict(torch.load('us_feature.pt',map_location=torch.device(device)))
    vgg11.to(device)
    # Put it into the Identity class defined before
    vgg11.classifier = Identity()
    vgg11.eval()
    # Get output features
    outputs = vgg11(input_datas)
    if device == 'cpu':
        outputs = torch.reshape(outputs,(outputs.shape[0],64,8,8)).numpy()
    else:
        outputs = torch.reshape(outputs,(outputs.shape[0],64,8,8)).cpu().detach().numpy()
    # Return extracted features
    return outputs

# DOT image preprocessing part
def DOT_image_preprocessing(n_depth, depth, z_radius, dot_image):
    # Get reconstructed start layer
    for d_layer in range(7):
        if n_depth > depth - z_radius:
            start_layer = d_layer;
            break
        else:
            n_depth = n_depth + 0.5;
    # Get DOT input from all reconstructed images
    dot_input = dot_image[:32,:32,start_layer:start_layer+3]
    # Permute it into Neural network input form
    dot_input = np.transpose(dot_input, [2,0,1])
    dot_input = dot_input[np.newaxis,:,:,:]
    # Convert the dot_input into tensor 
    dot_input = torch.tensor(dot_input, dtype=torch.float32, requires_grad=False)
    return dot_input

# Main function
def main_us_dot(us_image, dot_image, n_depth, depth, z_radius): 
    # DOT image preprocessing
    dot_input = DOT_image_preprocessing(n_depth, depth, z_radius, dot_image)
    # US feature extraction
    us_feature = us_feature_extract(us_image)
    # Convert into tensor
    us_feature = torch.tensor(us_feature, dtype=torch.float32, requires_grad=False)
    # Load DOT only model
    model_path = 'DOT_classification_reconImageOnly_used.pt'
    dot_model = torch.load(model_path)
    # Get DOT image features and DOT image prediction
    dot_pred, dot_feature = dot_model(dot_input)
    dot_pred = torch.sigmoid(dot_pred)
    # Combine DOT feature and US feature together
    input_features = torch.cat((dot_feature, us_feature), 1)
    # Load fusion model final stage weights
    model_path = 'DOT_classification_reconImage_withUS_finetune.pt'
    fusion_model = torch.load(model_path)
    # Get fusion model prediction
    final_pred, final_feature = fusion_model(input_features)
    final_pred = torch.sigmoid(final_pred)
    # Convert tenson into numpy array
    final_pred = final_pred.detach().numpy()
    # Return prediction
    return final_pred

# Unit test
if __name__ == "__main__":
    # Load input data
    mat = sio.loadmat('input_data')
    dot_image, us_image = mat['dot_image'], mat['us_image']
    n_depth, depth, z_radius = 0.5, 1.5, 0.99
    final_pred = main_us_dot(us_image, dot_image, n_depth, depth, z_radius)