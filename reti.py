import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import time

#transforming images into tensors
TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])
num_classes = 2
#Train parameters
num_epochs = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
TRAIN_DATA_PATH = "./a/"
TEST_DATA_PATH = "./f/"


# In[ ]:


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


# In[ ]:



    
#val_loss = train_CNN()    


# The torch.save() will the save the trained CNN in a file with extension .pth. This file can later be copied to some other low end computer to do predictions.

# In[ ]:





model = ConvNet()    
model.load_state_dict(torch.load('conv_net_model2_stop_class.pth', map_location = lambda storage, loc: storage))
model.eval()


# The Below set of lines are to pass one single image to CNN and print the result for it.

# In[ ]:


import os
from PIL import Image

img_rows, img_cols = 750, 750
def resize_image(path):
    im = Image.open(path)
    img = im.resize((750,750))
    img.save("re.jpg")
    
loader = TRANSFORM_IMG
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

resize_image("106.jpg")
classes = ["abinash","starnger"]

image = image_loader("re.jpg")
result = model(image)
x = result.data.max(1, keepdim=True)[1]
pred = (Variable(x).data).numpy() 
print(pred)

