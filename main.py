import cv2
import numpy as np
import os
import tensorflow as tf

def confusion_matrix(predictions, labels):
        
        matrix = np.zeros((2, 2))
        
        for i in range(len(predictions)):
            
            if predictions[i] > 0.5:
                prediction = 1
            else:
                prediction = 0
                
            matrix[prediction][labels[i]] += 1
            
        return matrix

def load_data(pathes):
    
    images = []
    labels = []
    apple_path, banana_path = pathes
    
    for i in os.listdir(apple_path):
        img = cv2.imread(apple_path + i)
        img = cv2.resize(img, (480, 480))/255
        images.append(img)
        labels.append(0)
        
    for i in os.listdir(banana_path):
        img = cv2.imread(banana_path + i)
        img = cv2.resize(img, (480, 480))/255
        images.append(img)
        labels.append(1)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def create_tensorflow_model():
    
    #initializing Sequential model
    model = tf.keras.models.Sequential()
    
    #add layers to the model. Conv2D and MaxPooling2D layers are used for feature extraction from images
    model.add(tf.keras.layers.Conv2D(64, (2, 2), input_shape = (480, 480, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(32, (2, 2)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    #Flatten layer is used to convert 2D data to 1D data
    model.add(tf.keras.layers.Flatten())
    
    
    #Dense layers are used for classification 
    model.add(tf.keras.layers.Dense(8, activation = "sigmoid"))
    model.add(tf.keras.layers.Dense(1, activation = "softmax"))
    
    
    #add regularization to each layer of the model to prevent overfitting
    for layer in model.layers:
        layer.kernel_regularizer = tf.keras.regularizers.l2(0.04)
    
    
    #compiling the model with binary_crossentropy loss function
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    
    return model

def main():
    
    #pathes for training data
    apple_path = 'data_training/Apple/'
    banana_path = 'data_training/Banana/'
    
    images, labels = load_data([apple_path, banana_path])
    
    if not os.path.exists('model.keras'):
    
        model = create_tensorflow_model()
        
        #training the model for 3 epochs, increase number if epochs for longer training
        model.fit(images, labels, epochs=5, shuffle=True)

        model.save('model.keras')
    
    else:
            
        model = tf.keras.models.load_model('model.keras')
    #pathes for testing data, should be different from training data for better evaluation
    
    apple_path = 'data_testing/Apple/'
    banana_path = 'data_testing/Banana/'
    
    images, labels = load_data([apple_path, banana_path])
    
    predictions = model.predict(images)
    
    matrix = confusion_matrix(predictions, labels)
    
    print(matrix)
    
    accuracy = (matrix[0][0] + matrix[1][1]) / np.sum(matrix)
    
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    main()