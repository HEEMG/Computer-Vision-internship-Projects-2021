# Convolutional Neural Network (CNN) Theory 


### Introduction 
A convolutional neural network (CNN) is class of artificial neural network (ANN) used in computer vision and processing which is specifically designed to process pixel data. Convolutional neural network (ConvNets) were first introduced in the 1980s by Yann LeCun, computer science researcher named as LeNet-5 to recognize hand-written numbers. CNN structure was first proposed by Fukushima in 1988.Alex Net, VGG, Google Net, Dense CNN and Fractal Net are generally considered the most popular architectures because of their performance in object recognition. ( Alom, et al., 2019) 
CNN is particularly used in image & video recognition, image analysis & classification, segmentation, media recreation, recommendation systems, brain-computer interfaces, natural language processing and financial time series.  ( Bhandare, et al., 2016) 

![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/0050c965-1a13-4060-b908-5f6feee47abd)


Fig 1: Schematic diagram of a basic CNN architecture (Source ( Phung & Rhee, 2019) 
CNN architecture 
CNN architecture consists two blocks feature extraction and classification. Feature extraction is also called convolution block which contains repetitions of a stack of convolution layers and Pooling layer. Similarly, classification block is also known as fully connected block where simple neural networks are connected to each other. (Yamashita, et al., 2018).The process in which input data are converted into output through these layers is called forward propagation as shown in figure. 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/e524ed4a-b0b2-4e17-bd8d-094a8215e76d)


Fig 2:  Convolution neural networks (CNN) architecture in image classification 
                                       Source:  Hacibeyoglu (2018) 
A. Features Extraction 
	i. 	Convolutional Layer 
A convolutional neural network is core part of CNN architecture. responsible for feature extraction performs linear and nonlinear operations such as convolution operation and activation function. The aim of Convolutional layer is to learn feature representations of the input. ( Guo, et al., 2017) Convolutional layer contains set of convolutional kernels or filters which gets convolved with the input image (N-dimensional metrics) to generate an output feature map. Filter is set of weights in a matrix applied on an image  or a matrix to obtain the required features. 
 
Let’s considered a 5 x 5 whose image pixel values are 0, 1 and filter matrix 3 x 3 as shown in below. 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/adac4c72-54a7-4dc4-b957-5ea7ff0b98ad)


Fig 3: 5×5 Image matrix (feature map) multiplies kernel or filter matrix 3×3(Source: ( Developers , 2021) 
 
Then the convolution of 5 x 5 image matrix multiplies with 3 x 3 filter matrix which is called “Feature Map” as output shown in below. 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/d979e235-a436-450f-b369-2f280a412356)


 Figure 4: 3 x 3 Output matrix Source: Developers (2021)      
 
ii. 	Non-Linearity (ReLU) 
ReLU has been used after every Convolution operation.  
The output is ƒ(x) = max(0,x).  


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/b0626fe5-35ea-4a4c-84bc-1abd1537d3f8)


Fig 5: ReLU function on Graph.  Source: Wu(2017) 
ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real-world data would want our ConvNet to learn would be non-negative linear values. There are other nonlinear functions such as tanh or sigmoid that can also be used instead of ReLU. 
On the following image we apply the Relu operation and replaces all the negative numbers by  0. 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/5d84dee5-fb44-45a9-aaad-c8559b4a82e8)

 Fig 6: ReLU Operation Source:   Hijazi, et al.(2015) & Prabhu (2018) 
 
iii. 	Pooling Layer: 
Pooling layer is second layer of CNN architecture. The pooling operation consists in reducing the size of the images while preserving their important characteristics. It summarizes value overall the value present. The purpose of the pooling layers is to reduce the spatial resolution of the feature maps and thus achieve spatial invariance to input distortions and translation. 
There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. Max Pooling also performs as a Noise Suppressant by discards the noisy activations altogether. Meanwhile; Average Pooling returns the average of all the values from the portion of the image as a noise suppressing mechanism covered by the Kernel as shown in figure. Thus, Max Pooling performs a lot better than Average Pooling. There are two important parameters in the pooling layer, one of which is the filter size and the second is the stride parameter.    ( Saha, 2018) 
 
 
![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/35b9dd51-8b89-4b90-a148-fb5583d8b0ab)

Figure 7: Comparison between Max and average pooling. Source:  ( Rawat & Wang, 2017)  Striding  
We use striding in such case when we do not want to capture all the information. So, we skip some neighboring cells. For example- 



![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/e520c1fa-fcfd-4fec-b71f-8a628cf3ee22)


Fig 7: Result of output image after striding  

Padding 
While applying convolutions, sometime we willnot get output equivalent to input because of loss of data over borders. So that, we append a border of zeros amd recalculate the convolutions covering all the input values as shown in below. 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/c6778091-59e9-420d-ba12-343a1bbdbca1)


Fig 8: Image matrix after padding and striding 
 
 
Classification Block iv. 	Fully connected Layer 
Fully-connected layers are the last part of every CNN architecture where each neuron inside a layer is connected with each neuron from its previous layer also known as dense layers. The last layer of Fully-Connected layers is used as the classifier of output image of the CNN architecture. The final fully connected layer typically has the same number of output nodes as the number of classes. Fully-Connected Layers are feed-forward artificial neural network (ANN) works under the principle of traditional multi-layer perceptron neural network (MLP).  
The FC layers take input as feature maps from the final convolutional or pooling layer in the form of a set of metrics and those metrics are flattened to create a vector and this vector is then pass into the FC layer to generate the final output of CNN. ( Ghosh, et al., 2020). Flattening is the function that converts the pooled feature map to a single column that is passed to the fully connected layer. Finally, we use an activation function such as SoftMax or sigmoid to classify the outputs. 

![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/f65d4978-bb5a-40cc-91ea-1d2763c668cd)


Fig 9: Flattening of a 3x3 image matrix into a 9x1 vector 

![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/8ecd77bc-f0e4-47d0-8f8e-44e7ec0a9ef9)


Fig 10: Example of fully-connected neural network (Source: (Pelletier, et al., 2019) 

### Conclusion 

In conclusion, Convolutional Neural Network (CNN) consists convolution layer and fully connected layer. The convolution layer always contains two basic functions, namely convolution and pooling. The convolution operation using multiple filters generate feature map by extracting features from the data set without losing important features. The pooling operation, also called subsampling, is used to reduce the dimensionality of feature maps from the convolution operation. These features are passed to the fully connected layer which consists of activation function and classify the image. The detail diagram of CNN is listed following as shown in figure. 

![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/3ffe41f7-470a-4fd5-a15d-6cdde4934a48)


Fig11: Convolutional Neural Networks (CNN) architectures presentation 
(Source: https://dev.to/afrozchakure/cnn-in-a-brief-27gg) 


### Drawback of Convolutional Neural Network (CNN) 

CNN has many limitations on its applications. It makes prediction and classifies based on certain components are present in that image. One of them is its inability to encode the position and orientation of object. Let’s consider our face where have the face oval, two eyes, a nose and a mouth. It’s enough for CNN in having presence of these components can be a very strong indicator to consider that there is a face in the image. Orientational and relative spatial relationships between these components were not considered for CNN. ( Pechyonkin, 2017)The example is given below.  


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/22f24d44-1eab-4e3c-ab8b-d14b2dbb4a30)


Figure 12: Both pictures are similar to CNN, since they both contain similar components. Source:   Pechyonkin(2017) 

#### Reference : https://crawling-sole-80f.notion.site/CONVOLUTION-NEURAL-NETWORK-39aa64b0ddce445f98f8de129cc90ba6


# Object Detection: YOLO and SDD 
 
### YOLO (You only look Once) 
* Yolo was created by Joseph Redmon and Ali Farhadi in 2016.Yolo is based on Google Net. 
* Uses single convolutional network predicts the bounding boxes and the class probabilities. 
* Take an image and fragments into a S* S grid within each of the grid we take n bonding box. Usually we used 17*17 grid size. 
* The network ouputs a class probability and off values for the bounding box for each of the bounding box. 
* The bounding box which have class probability above a threshold value is selected and use to localize the object within the image. Finally, Non-Max Suppression and IOU are used to eliminate overlapping boxes. 
* Superfast 45 frames per second and accuracy 80.3%. 
*  Yolo is difficulty for detection with small objects that appear in groups, such as flocks of birds. The main source of error is incorrect localizations.  


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/c6c0ee78-7a48-4087-913e-1f7d92cba7dc)


Fig: Workflow of Yolo model 

       	 
The final prediction of image S×S×(5B+C) is produced by two fully connected layers over the whole conv feature map where B bounding boxes, confidence for those boxes, and C class probabilities 


![image](https://github.com/Hem5555/Computer-Vision-internship-Projects-2021/assets/121716939/cc327aab-0768-4a28-9eb8-c2d8fe75328f)


Fig: The network architecture of YOLO 
### Some Facts about YOLO Version 
* Yolo tiny 3 is 2/3rd    times accurate and 8 times frame rate per second (FPS) than YOLOV4. 
* Fast YOLO is a fast version of YOLO, which uses 9 convolutional layers instead of 24. It is faster than YOLO but has lower mAP 
* YoloV3 uses a variant of Darknet architecture and has 53 layers training with ImageNet dataset 
* YoloV4 architecture is made of  CSPDarkness53, spatial pyramid pooling, additional module PANet path aggregation neck and yolov3head. 
* Yolov3 is better and faster than SSD and worse than RetinaNet but 3.8Xfaster. 
* Yolov5 was released   by a company Ultralytics in 2020.  
 
### SSD(Single Shot Detector) 
* SSD work or modification of the architecture of the VGG-16.  
* SSD decarded the use of the fully connected layers.  
* SSD has two components: Backbone Model and SSD Model  
− Backbone Model is predefined image classification network as extract feature maps.  
− SSD Model is the convolution filter added to backbone model where the output are interpreted as the bounding boxes and classes of the objects in the spatial location of the final filter’s activations.  
* SSD is more efficient and has a good accuracy.  
* Instead of sliding window SSD divides the images using the grid where each grid cell is responsible for the detecting objects in the of region of the image.  
Here, detecting object is predicting the class and location of an object with in that region. If no object is present, we consider it as background where that location is ignored. 
* SSD is uses non-maximum suppression to remove duplicate predictions pointing to same objects.  
* Anchor box is used to detect the multiple objects in an image. It is simple boxes assigned with multiple prior boxes which are predefined and have fixed size and shape within the  grid cell.  
* MultiBox’s loss function is combined of two critical components Confidence Loss and Location Loss.   
− Confidence Loss measures how confident the network is of the object of the computed bounding box where categorical cross-entropy is used to compute this loss.  
− Location Loss measures how far away the network’s predicted bounding boxes are from the ground truth ones from the training set.  
− FORMULA:  
Multibox_loss = Confidence loss + alpha * Location_loss  
Where, the term alpha helps in balancing the contribution of the location  loss. 
 







