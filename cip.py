import cv2 
import os
from PIL import Image
import sys
import dlib
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import time
TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
num_classes = 2
classes = ["abinash","starnger"]	
crop_width = 108
img_rows, img_cols = 750, 750
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 75, kernel_size=10, stride=10, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2))
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(75, 75, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(75, 75, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out = nn.Dropout()    
        self.fc1 = nn.Linear(6 * 6 * 75, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, num_classes) 
            
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 6 * 6 * 75)
        out = self.drop_out(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

model = ConvNet()    
model.load_state_dict(torch.load('conv_net_model2_stop_class.pth', map_location = lambda storage, loc: storage))
model.eval()
def resize_image(path):
    im = Image.open(path)
    img = im.resize((750,750))
    img.save("r.jpg")
loader = TRANSFORM_IMG
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def valid(img) :
	resize_image("1.jpg")
	classes = ["abinash","starnger"]
	image = image_loader("r.jpg")
	result = model(image)
	x = result.data.max(1, keepdim=True)[1]
	pred = (Variable(x).data).numpy() 
	print(pred)
def FrameCapture(): 
    # Path to video file 
    vidObj = cv2.VideoCapture(0) 
    # Used as counter variable 
    # Function to extract frames 
    success = 1  
    count = 0
    filename_inc =1
    simple_crop = False
    single_crop = False
    success, image = vidObj.read() 
    while success:  
        # checks whether frames were extracted 
        success, image = vidObj.read() 
        
        cv2.imshow('frame',image)
        # Saves the frames with frame-count 
        cv2.imwrite("frame.jpg", image) 
        count += 1
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(image, 1)
        print(" %d detected faces" % (len(detected_faces)))
        
        for i, face_rect in enumerate(detected_faces):
            width = face_rect.right() - face_rect.left()
            height = face_rect.bottom() - face_rect.top()
            new_face_rect=dlib.rectangle(0,0,width,height)
            if width >= crop_width and height >= crop_width:
                image_to_crop = Image.open("frame.jpg")
                if single_crop:
                    crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
                else:
                    size_array = []
                    size_array.append(face_rect.top())
                    size_array.append(image_to_crop.height - face_rect.bottom())
                    size_array.append(face_rect.left())
                    size_array.append(image_to_crop.width - face_rect.right())
                    size_array.sort()
                    short_side = size_array[0]
                    crop_area = (face_rect.left() - size_array[0] , face_rect.top() - size_array[0], face_rect.right() + size_array[0],         face_rect.bottom() + size_array[0])
                he,wi, c= image.shape
                W1=wi//2-wi//6
                W2=wi//2+wi//6
                x=face_rect.left()
                w=width
                x1=x+w
                if (W1<x and W2>x1) or (W1>x and W2<x1) or (x-W1+w>=w//2 and W2>x1) or (W2-x1+w>=w//2 and W1<x) :
                   print("Person at center focusing straight mode")
                elif (W1>x1 and W1>x) or (W1>x and x1-W1+w>=w//2)  :
                   print("Preson at Left turning left mode")
                else : 
                    print("Person at Right turning right mode")
                cropped_image = image_to_crop.crop(crop_area)
                crop_size = (crop_width, crop_width)
                cropped_image.thumbnail(crop_size)
                cropped_image.save(str(filename_inc) + ".jpg", "JPEG")
                valid(cropped_image)



        if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
if __name__ == '__main__': 
  

    # Calling the function 
    FrameCapture()
