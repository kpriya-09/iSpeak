# iSpeak
The project is designed to capture data from the webcam and predict and display the sign on the screen. The user would be signing an ASL Alphabet and the Region of Interest is captured and then processed before sending it to the Machine Learning model. The processing part includes, reshaping, resizing, gray-scaling and blurring the input according to the input data specifications of the model.


The project uses Kaggle Sign Language MNIST dataset.  It follows CSV format with labels and pixel values in single rows.  Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255.


The Machine Learning Model uses Convolutional Neural Network to predict the processed input image and the prediction is sent back to the Computer Vision Model which displays it on the screen. The image is passed through different layers in the CNN model which extract the most significant features in order to provide maximum accuracy.
