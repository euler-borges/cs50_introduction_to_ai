# Process

## First try

Used the same model as in the class (with small tweaks, adapting to the problem).

```python
def get_model():
"""
Returns a compiled convolutional neural network model. Assume that the
`input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
The output layer should have `NUM_CATEGORIES` units, one for each category.
"""

    model = tf.keras.models.Sequential([
        # Convolutional input layer. 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

Obtaining the following results from the training:

```bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.0910 - loss: 3.9635
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.1557 - loss: 3.0877
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.2140 - loss: 2.7536
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.2620 - loss: 2.5353
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.2904 - loss: 2.3922
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.3179 - loss: 2.2156
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.3520 - loss: 2.0971
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.3863 - loss: 1.9501
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.4062 - loss: 1.8802
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.4357 - loss: 1.7757
333/333 - 1s - 2ms/step - accuracy: 0.6319 - loss: 1.0768
```

Results were not ideal, as 43% and 63% are way too low.

## Second try

Decided to double the amount of hidden neurons.

```python
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional input layer. 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

```bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0512 - loss: 6.4174
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - accuracy: 0.0564 - loss: 3.5921
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5424
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5191
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5086
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5039
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5013
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0593 - loss: 3.5002
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.0593 - loss: 3.4995
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.0593 - loss: 3.4992
333/333 - 1s - 2ms/step - accuracy: 0.0518 - loss: 3.4973
```

Results got way worse, not the best approach.

## Third try

Tried with 2 hidden layers.

```python
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional input layer. 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

```bash
Skipping registering GPU devices...
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.0533 - loss: 4.6230
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0541 - loss: 3.5238
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0544 - loss: 3.5118
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 5s 10ms/step - accuracy: 0.0506 - loss: 3.5123
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0532 - loss: 3.5100
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0535 - loss: 3.5068
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.0551 - loss: 3.5066
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0541 - loss: 3.5050
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.0529 - loss: 3.5054
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.0546 - loss: 3.5044
333/333 - 1s - 2ms/step - accuracy: 0.0575 - loss: 3.4990
```

Still, bad results, first try being the best so far.

## Fourth try

Decided to add a second convolutional layer.

```python
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional input layer. 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer. 64 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

```bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.0622 - loss: 4.2765
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.1891 - loss: 3.1170
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.4725 - loss: 1.7965
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.6438 - loss: 1.1525
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.7337 - loss: 0.8488
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 6s 11ms/step - accuracy: 0.7903 - loss: 0.6585
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - accuracy: 0.8266 - loss: 0.5520
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.8527 - loss: 0.4719
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.8750 - loss: 0.4154
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 5ms/step - accuracy: 0.8873 - loss: 0.3695
333/333 - 1s - 2ms/step - accuracy: 0.9652 - loss: 0.1286
```

Best results so far, 88% and 96% are very good.

## Fifth try

As adding convolitional layers was what was giving the best results, I decided to add another one, with 128 filters, to test if it would get any better.

```python
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional input layer. 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer. 64 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer. 128 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
```

```bash
Epoch 1/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.2572 - loss: 3.1288
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.5974 - loss: 1.3596
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.7649 - loss: 0.7751
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.8421 - loss: 0.5265
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.8854 - loss: 0.3798
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9028 - loss: 0.3217
Epoch 7/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9281 - loss: 0.2527
Epoch 8/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9339 - loss: 0.2336
Epoch 9/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.9442 - loss: 0.1925
Epoch 10/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 7s 13ms/step - accuracy: 0.9464 - loss: 0.1898
333/333 - 1s - 3ms/step - accuracy: 0.9540 - loss: 0.1816
```

The results were very good, with 94.6% on the training data and 95.4% on the testing data. Good results, but the difference from last try was probably just a difference at the rando starting weights.
