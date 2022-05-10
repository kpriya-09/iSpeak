# iSpeak
The project is designed to capture data from the webcam and predict and display the sign on the screen. 
The user would be signing an ASL Alphabet and the Region of Interest is captured and then processed before sending it to the Machine Learning model. 
The processing part includes, reshaping, resizing, gray-scaling and blurring the input according to the input data specifications of the model.


The project uses Kaggle Sign Language MNIST dataset. 

https://www.kaggle.com/datasets/datamunge/sign-language-mnist


The Machine Learning Model uses Convolutional Neural Network to predict the processed input image and the prediction is sent back to the Computer Vision Model which displays it on the screen. 
The image is passed through different layers in the CNN model which extract the most significant features in order to provide maximum accuracy.
