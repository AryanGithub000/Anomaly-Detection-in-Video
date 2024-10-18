# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM, BatchNormalization, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score

# Directories for train, val, and test data
train_folder = 'Data_Split/train'
val_folder = 'Data_Split/val'

# Image size and batch size
img_size = (112, 112)  # Reduced image size
batch_size = 8

# Data generators with rescaling for both training and validation data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directories
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

# Define a custom callback to calculate precision, recall, and F1 score after each epoch
class F1Metrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_predict = (self.model.predict(val_generator) > 0.5).astype(int)
        val_targ = val_generator.classes  # Ground truth labels
        _precision = precision_score(val_targ, val_predict)
        _recall = recall_score(val_targ, val_predict)
        _f1 = f1_score(val_targ, val_predict)
        print(f"— val_precision: {_precision:.4f} — val_recall: {_recall:.4f} — val_f1: {_f1:.4f}")

# CNN + LSTM Model
def build_cnn_lstm_model(input_shape):
    model = Sequential()

    # TimeDistributed CNN layers
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))
    
    # Flattening the CNN output
    model.add(TimeDistributed(Flatten()))

    # Reshape to feed into LSTM
    model.add(Reshape((-1, 32 * 28 * 28)))  # Adjust based on the feature map size after the CNN
    
    # LSTM layer with fewer units
    model.add(LSTM(32, return_sequences=False))

    # Dense and output layer
    model.add(Dense(32, activation='relu'))  # Reduced dense layer units
    model.add(Dropout(0.5))

    # Final output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))  #Added 2 for our

    return model

# Set input shape for grayscale images (112x112 size)
input_shape = (None, 112, 112, 1)

# Build the model
model = build_cnn_lstm_model(input_shape)

# Compile the model with precision and recall as metrics
# Guides the model for better improvement 
# Adjusts internal settings based on learning rates
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall()])

# Model summary
model.summary()

# Train the model with the F1Metrics callback
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1, #epochs changed
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[F1Metrics()]  # Custom callback for precision, recall, and F1 score
)

# Save the model
# Can be used for metrics analyze
model.save('cnn_lstm_model_optimized_reduced.h5')

print("Training complete. Model saved.")
