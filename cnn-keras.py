# Convolutional Neural Network with Keras

# Import the libraries and dataset
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist

# Get the MNIST Data and split it into the training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

# Normalize the data so that each dimension has approximately the same scale
x_train /= 255
x_test /= 255

# Encode the data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model to the training data
model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_data=(x_test, y_test))

# Evaluate the accuracy of the model
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test accuracy', score[1])