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
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5] )
    ])
num_classes = 2
#Train parameters
num_epochs = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005


TRAIN_DATA_PATH = "face"
TEST_DATA_PATH = "test"

#loading the image data
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=1)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0))
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out = nn.Dropout()    
        self.fc1 = nn.Linear(121*8,256)
        nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256,256)
        nn.ReLU(inplace=True),
        self.fc3 = nn.Linear(256, 2) 
            
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1,8*121)
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

def train_CNN():
    
    print("Number of train samples: ", len(train_data))
    print("Number of test samples: ", len(test_data))
    print("Detected Classes are: ", train_data.class_to_idx) # classes are detected by folder structure

    model = ConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    # Training and Testing
    total_step = len(train_data_loader)
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):
          
         b_x = Variable(images)
         b_y = Variable(labels)
         output = model(b_x)
         loss = criterion(output, b_y)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         print('Step [{}/{}], Loss: {:.9f}'.format(i + 1, total_step, loss.data))
        print('Epoch [{}/{}], Loss: {:.9f},Took time :{:.4f}s'.format(epoch+1,num_epochs,loss.data,time.time()-start_time))
        train_loss.append(loss.data)
        total_val_loss = 0
        for inputs, labels in test_data_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = model(inputs)
            val_loss_size = criterion(val_outputs, labels)
            total_val_loss += val_loss_size.data
            
        print("Epoch [{}/{}],Validation loss = {:.9f}".format(epoch+1,num_epochs,total_val_loss / len(test_data_loader)))
        val_loss.append(total_val_loss/len(test_data_loader))
        start_time = time.time()
        
    #Train accuracy
    model.eval()
    correct =0
    for data, target in train_data_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()        
    print('Train accuracy = {:.4f}'.format(correct /len(train_data)))
    
    #Test accuracy
    model.eval()
    correct =0
    for data, target in test_data_loader:
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()        
    print('Test accuracy = {:.4f}'.format(correct /len(test_data)))
    return val_loss
    
val_loss = train_CNN()    


# The torch.save() will the save the trained CNN in a file with extension .pth. This file can later be copied to some other low end computer to do predictions.

# In[ ]:


model = ConvNet()    
torch.save(model.state_dict(),'conv_net_model2_stop_class.pth')


# The model is saved and loaded again just for checking. model.load_sate_dict() is using to load the parameter file.

# In[ ]:


model = ConvNet()    
model.load_state_dict(torch.load('conv_net_model2_stop_class.pth', map_location = lambda storage, loc: storage))
model.eval()


# The Below set of lines are to pass one single image to CNN and print the result for it.

# In[ ]:


import os
from PIL import Image

img_rows, img_cols = 100, 100
def resize_image(path):
    im = Image.open(path)
    img = im.resize((108,108))
    img.save("resizedImage.jpg")
    
loader = TRANSFORM_IMG
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

resize_image("212.jpg")
classes = ["Forward","Left"]

image = image_loader("resizedImage.jpg")
result = model(image)
x = result.data.max(1, keepdim=True)[1]
pred = (Variable(x).data).numpy() 
print(pred)


