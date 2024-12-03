#Requirement Library
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # If you want utilize GPU, uncomment this line
from sklearn.utils import shuffle
import rosbag
import time
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import Adam

# Check GPU availability - You don't need a gpu to train this model
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpu_available = tf.test.is_gpu_available()
print('GPU AVAILABLE:', gpu_available)

#========================================================
# Functions
#========================================================
#Linear maping
def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping function."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

#Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    mean_loss = np.mean(loss)
    return mean_loss
#========================================================
# Global Data
#========================================================

# Initialize lists for data
lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []
model_name = 'TLN_temporal_info'
model_files = [
    './Models/'+model_name+'_noquantized.tflite',
    './Models/'+model_name+'_int8.tflite'
]
dataset_path = [
    # './Dataset/out.bag', 
    './Dataset/out1.bag', 
    './Dataset/out2.bag', 
]
loss_figure_path = './Figures/loss_curve.png'
down_sample_param = 2 # Down-sample Lidar data
lr = 5e-5
loss_function = 'huber'
batch_size = 64
num_epochs = 20
hz = 40
temporal_length = 1

# Initialize variables for min and max speed
max_speed = 0
min_speed = 0

#========================================================
# Get Dataset
#========================================================

# Iterate through bag files
for pth in dataset_path:
    if not os.path.exists(pth):
        print(f"out.bag doesn't exist in {pth}")
        exit(0)
    good_bag = rosbag.Bag(pth)

    lidar_data = []
    servo_data = []
    speed_data = []

    # Read messages from bag file
    for topic, msg, t in good_bag.read_messages():
        if topic == 'Lidar':
            ranges = msg.ranges[::down_sample_param]
            lidar_data.append(ranges)
        if topic == 'Ackermann':
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            
            servo_data.append(data)
            if s_data > max_speed:
                max_speed = s_data
            speed_data.append(s_data)

    # Convert data to arrays
    lidar_data = np.array(lidar_data) 
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)

    # Shuffle data
    shuffled_data = shuffle(np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1), random_state=62)
    shuffled_lidar_data = shuffle(lidar_data, random_state=62)

    # Split data into train and test sets
    train_ratio = 0.85
    train_samples = int(train_ratio * len(shuffled_lidar_data))
    x_train_bag, x_test_bag = shuffled_lidar_data[:train_samples], shuffled_lidar_data[train_samples:]

    # Extract servo and speed values
    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]

    # Extend lists with train and test data
    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])
    speed.extend(y_train_bag[:, 1])

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])
    test_speed.extend(y_test_bag[:, 1])

    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, Servo: {len(test_servo)}, Speed: {len(test_speed)}')

# Calculate total number of samples
total_number_samples = len(lidar)

print(f'Overall Samples = {total_number_samples}')
lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
speed = linear_map(speed, min_speed, max_speed, 0, 1)
test_lidar = np.asarray(test_lidar)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(lidar)} Training samples ---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
print(f'Loaded {len(test_lidar)} Testing samples ---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

# Check array shapes
assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)

#======================================================
# Adding Temporal Information
#======================================================

# Initialize new lists for processed data
lidar_sequences = []
servo_averages = []
speed_averages = []

# Process training data
for i in range(len(lidar) - temporal_length + 1):
    # Stack 5 LiDAR scans
    sequence = np.stack(lidar[i:i + temporal_length], axis=0)  # Shape: (5, lidar_dim)
    lidar_sequences.append(sequence)
    
    # Calculate average servo and speed values
    # Determine avg_servo based on the majority sign of servo values
    buffer_servo = servo[i:i + temporal_length]
    if np.sum(buffer_servo > 0) > np.sum(buffer_servo < 0):
        avg_servo = np.max(buffer_servo)  # Majority positive, take max
    else:
        avg_servo = np.min(buffer_servo)  # Majority negative, take min
    
    avg_speed = np.max(speed[i:i + temporal_length])
    servo_averages.append(avg_servo)
    speed_averages.append(avg_speed)

# Convert to numpy arrays
lidar_sequences = np.array(lidar_sequences)  # Shape: (num_sequences, 5, lidar_dim)
servo_averages = np.array(servo_averages)    # Shape: (num_sequences,)
speed_averages = np.array(speed_averages)    # Shape: (num_sequences,)

# Print updated shapes
print(f'Processed Training Samples: {lidar_sequences.shape[0]} sequences')
print(f'Lidar Sequence Shape: {lidar_sequences.shape}')
print(f'Servo Shape: {servo_averages.shape}')
print(f'Speed Shape: {speed_averages.shape}')

# Initialize new test_lists for processed data
test_lidar_sequences = []
test_servo_averages = []
test_speed_averages = []

# Process training data
for i in range(len(test_lidar) - temporal_length + 1):
    # Stack 5 LiDAR scans
    sequence = np.stack(lidar[i:i + temporal_length], axis=0)  # Shape: (5, lidar_dim)
    test_lidar_sequences.append(sequence)
    
    # Determine avg_servo based on the majority sign of servo values
    buffer_servo = servo[i:i + temporal_length]
    if np.sum(buffer_servo > 0) > np.sum(buffer_servo < 0):
        avg_servo = np.max(buffer_servo)  # Majority positive, take max
    else:
        avg_servo = np.min(buffer_servo)  # Majority negative, take min
    avg_speed = np.max(speed[i:i + temporal_length])
    test_servo_averages.append(avg_servo)
    test_speed_averages.append(avg_speed)

# Convert to numpy arrays
test_lidar_sequences = np.array(test_lidar_sequences)  # Shape: (num_sequences, 5, lidar_dim)
test_servo_averages = np.array(test_servo_averages)    # Shape: (num_sequences,)
test_speed_averages = np.array(test_speed_averages)    # Shape: (num_sequences,)

# Print updated shapes
print(f'Processed Training Samples: {test_lidar_sequences.shape[0]} sequences')
print(f'Lidar Sequence Shape: {test_lidar_sequences.shape}')
print(f'Servo Shape: {test_servo_averages.shape}')
print(f'Speed Shape: {test_speed_averages.shape}')

# Check array shapes after processing
assert len(lidar_sequences) == len(servo_averages) == len(speed_averages)
assert len(test_lidar_sequences) == len(test_servo_averages) == len(test_speed_averages)
#======================================================
# Split Dataset
#======================================================

print('Splitting Data into Train/Test')
train_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data =  np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)
train_sequences_data = np.concatenate((servo_averages[:, np.newaxis], speed_averages[:, np.newaxis]), axis=1)
test_sequences_data =  np.concatenate((test_servo_averages[:, np.newaxis], test_speed_averages[:, np.newaxis]), axis=1)

# Check array shapes
print(f'Train Data(lidar): {lidar.shape}')
print(f'Train Data(servo, speed): {servo.shape}, {speed.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_servo.shape}, {test_speed.shape}')

#======================================================
# DNN Arch
#======================================================

num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=24, kernel_size=(temporal_length, 10), strides=(1, 4), activation='relu', 
                           input_shape=(temporal_length, num_lidar_range_values, 1)),  # (5, 540, 1)
    tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 8), strides=(1, 4), activation='relu'),
    tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 4), strides=(1, 2), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh')  # Predict normalized steering and speed
])
#======================================================
# Model Compilation
#======================================================


optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss=loss_function)
print(model.summary())

#======================================================
# Model Fit
#======================================================
start_time = time.time()
history = model.fit(lidar_sequences, np.concatenate((servo_averages[:, np.newaxis], speed_averages[:, np.newaxis]), axis=1),
                    epochs=num_epochs, batch_size=batch_size, validation_data=(test_lidar_sequences, test_sequences_data))

print(f'=============>{int(time.time() - start_time)} seconds<=============')

# Plot training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# Model Evaluation
#======================================================

print("==========================================")
print("Model Evaluation")
print("==========================================")

# Evaluate test loss
test_loss = model.evaluate(test_lidar_sequences, test_sequences_data)
print(f'Overall Test Loss = {test_loss}')

# Calculate and print overall evaluation
y_pred = model.predict(test_lidar_sequences)
hl = huber_loss(test_sequences_data, y_pred)
print('\nOverall Evaluation:')
print(f'Overall Huber Loss: {hl:.3f}')

# Calculate and print speed evaluation
# speed_y_pred = model.predict(test_sequences_data)[:, 1]
# speed_test_loss = huber_loss(test_sequences_data[:, 1], speed_y_pred)
# print("\nSpeed Evaluation:")
# print(f"Speed Test Loss: {speed_test_loss}")

# # Calculate and print servo evaluation
# servo_y_pred = model.predict(test_lidar_sequences)[:, 0]
# servo_test_loss = huber_loss(test_sequences_data[:, 0], servo_y_pred)
# print("\nServo Evaluation:")
# print(f"Servo Test Loss: {servo_test_loss}")

#======================================================
# Save Model
#======================================================
# Save non-quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_path = './Models/' + model_name + "_noquantized.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
    print(f"{model_name}_noquantized.tflite is saved.")

# # Save int8 quantized model
# rep_32 = lidar.astype(np.float32)
# rep_32 = np.expand_dims(rep_32, -1)
# dataset = tf.data.Dataset.from_tensor_slices(rep_32)

# def representative_data_gen():
#     for input_value in dataset.batch(len(lidar)).take(rep_32.shape[0]):
#         yield [input_value]

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# quantized_tflite_model = converter.convert()

# tflite_model_path = './Models/' + model_name + "_int8.tflite"
# with open(tflite_model_path, 'wb') as f:
#     f.write(quantized_tflite_model)
#     print(f"{model_name}_int8.tflite is saved.")

# print('Tf_lite Models also saved')

#======================================================
# Evaluated TfLite Model
#======================================================

