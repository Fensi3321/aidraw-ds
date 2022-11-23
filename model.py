import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, train_y = map(lambda x: x[..., None], [train_x, train_y])

encoder = OneHotEncoder(categories=[range(10)])
encoded_train_y, endoced_test_y  = map(lambda y: encoder.fit_transform(y.reshape(-1, 1)).toarray(), [train_y, test_y])
encoded_test_y = encoder.fit_transform(test_y.reshape(-1, 1)).toarray()


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(train_x.shape[1:]),
    tf.keras.layers.Conv2D(5, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(encoded_train_y.shape[-1], activation='softmax')
])

adam_optimizer =  tf.keras.optimizers.Adam()
model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(train_x, encoded_train_y, batch_size=128, epochs=5, validation_data=[test_x, encoded_test_y])

model.save('my_model')

