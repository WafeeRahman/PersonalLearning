#Object Detection using CNN's and Single Shot Multibox Detection


#Imports

import torch # Utilizing Pytorch to use and leverage backpropogation in CNN's within SSD Algortihhm
from torch.autograd import Variable #Use the variable class to convert tensors to tensor - gradient object (Torch Variable)
import cv2 # Draw Rectangles Around Video

#Base Transform will transform images to be compatible with the Neural Network
#VOC Classes will map classifications for different objects

from data import BaseTransform, VOC_CLASSES as labelmap

#Used Build SSD to construct SSD Object
from ssd import build_ssd

#Process Video Frames through Image IO
import imageio

"""
Detection Function to Detect Objects within Video

Frame - This function will work on each frame of the inputted video as passed by imageio, detecting objects through SSD

Net - Neural Network used to detect objects

Transform - Transformation of images from imageio so that the neural network can work with it. 

Return Val: The same frame but with rectangles that detect objects, and their label classification using SSD

""" 

#Since we aren't using openCV, we won't need a grayscale image for detection. 
def detect(frame, net, transform):
    #Take the height and width, the shape also returns the colour channel, but we arent interested in it
    height, width  = frame.shape[:2]

    #Transform the input frame into a torch variable to pass into the SSD algorithm
    #Before we can get this torch variable, we need to do four sequential transformations

    # 1. Apply the tranform function so that we can pass the frame into the neural network
    frame_transform =  transform(frame)[0] #Returns multiple values, the transform is the first one

    # 2. Transform frame_transform from numpy array to a pytorch tensor using the from_numpy function
    # 3. Transform colour channel from RGB to BGR to match our trained neural network
    # 4. Add a fake dimension to the tensor to create a fake batch input to the neural network, as the neural network accepts batch data using the unsqueeze function
    
    x = torch.from_numpy(frame_transform).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 300, 300]
    
    # 5. Convert this batch of torch tensors in to a torch variable
    x = Variable(x)

    #Place Torch Variable into the neural network
    y = net(x)

    #Take the tensors of the predicted objects returned by neural network
    detections = y.data

    #Create a new scaling tensor object of two widths and heights
    #We need to normalize the position of the detected objects using this tensor
    
    #The First width and height will capture to top right and second will capture bottom right of detected object
    scale = torch.Tensor([width, height, width, height])

    #Tensor Structure
    #Detections = [Batch, number of classifications, number of occurrences, (score, x0, y0, x1, y1)]
    #Score determines if a class is present on the image, x0,y0 is the upper left and x1,y1 is the lower right corner for the box

    #Dectections.size(1) is the number of classes
    for i in range(detections.size(1)):
        j = 0 #Occurence
        
        #detections[0, i, j, 0] - Check the score of the occurence j ofr the class i
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale) #Take the coordinates of upperleft and lowerright if we have a probable score and normalize using our scale
            pt = pt.numpy() #Convert to np array to pass into openCV

            #Place rectangle around detected object
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])) , (int(pt[2]), int(pt[3])), (255,0,0), 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (2555, 255, 255), 2, cv2.LINE_AA)
            j += 1 #Check the next occurrence of class i
    return frame    

#Build SSD Neural Network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location= lambda storage, loc: storage))

#Creating the Transformation to transform image to be compatible to the neural network
#Net Size to accept images into neural net, and 3 scale factors
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Apply Object Transformation on Video
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)

writer.close()
