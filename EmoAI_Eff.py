import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, BatchNormalization, Input, 
                                     GlobalAveragePooling2D, Add, Activation,
                                     SeparableConv2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("../fer2013.csv/fer2013.csv")

# Data Preprocessing
def preprocess_data(data):
    X = np.array([np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48, 1) 
                  for pixels in data['pixels']]) / 255.0  
    y = to_categorical(data['emotion'], num_classes=7)
    return X, y

# Split data
train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

X_train, y_train = preprocess_data(train_data)
X_val, y_val = preprocess_data(val_data)
X_test, y_test = preprocess_data(test_data)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Improved Residual Block
# Improved Residual Block with 1x1 Conv for Dimension Matching
def residual_block(x, filters):
    shortcut = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal')(x)  # 1x1 Conv for matching dimensions
    x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


# Model Architecture
input_layer = Input(shape=(48, 48, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = residual_block(x, 128)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = SeparableConv2D(
    256, 
    (3, 3), 
    activation='relu', 
    padding='same', 
    depthwise_initializer='he_normal',   # Correct initializer for SeparableConv2D
    pointwise_initializer='he_normal'    # Ensures consistency in both depthwise & pointwise layers
)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

output_layer = Dense(7, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Optimizer with Warmup Strategy
initial_learning_rate = 0.0005
optimizer = Adam(learning_rate=initial_learning_rate)

# Adaptive Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Class Weights to Handle Imbalance
class_weights = {0: 1.5, 1: 1.0, 2: 1.2, 3: 1.1, 4: 1.3, 5: 1.4, 6: 1.0}

# Model Training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=25,
    callbacks=[lr_scheduler, early_stopping],
    class_weight=class_weights
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the improved model
model.save('optimized_emotion_detection_model.keras')

# Plotting Accuracy Curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
