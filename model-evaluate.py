model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, 
          batch_size = 128, 
          epochs = 6, 
          verbose = 1, 
          validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test accuracy', score[1])
