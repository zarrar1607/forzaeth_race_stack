import time
import numpy as np
import rospy
import tensorflow as tf
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

# ROS topics
lid = '/scan'  # Lidar topic
drive = '/vesc/high_level/ackermann_cmd_mux/input/nav_1'

# Global variables
prev = 0  # Previous button state
curr = 0  # Current button state
start_position = None  # Start position for calculating distance traveled
total_distance = 0.0  # Total distance traveled
lidar_data = None  # Placeholder for Lidar data
model_name = './Models/TLN_noquantized.tflite'
subsample_lidar = 2  # Down-sample Lidar data: skipping lidar scan by subsample_lidar
rospy.init_node('Autonomous')  # ROS initialization
hz = 40  # Frequency (Hz)\
rate = rospy.Rate(hz)  # Rate controller
period = 1.0 / hz  # Time period

# Callback to receive Lidar data


def callback(l):
    global lidar_data
    ldata = l.ranges[::subsample_lidar]
    # Reshape and convert to float32
    ldata = np.expand_dims(ldata, axis=-1).astype(np.float32)
    ldata = np.expand_dims(ldata, axis=0)  # Add batch dimension
    lidar_data = ldata  # Store the processed Lidar data

# Callback to calculate distance traveled


def odom_callback(msg):
    global start_position, total_distance
    if start_position is None:
        start_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        return
    current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    distance = np.linalg.norm(
        np.array(current_position) - np.array(start_position))
    total_distance += distance
    start_position = current_position

# Load the TensorFlow Lite model


def load_model():
    global interpreter, input_index, output_details
    interpreter = tf.lite.Interpreter(model_path=model_name)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_details = interpreter.get_output_details()

# Run the Lidar data through the model and get servo and speed predictions


def dnn_output():
    global lidar_data, inf_time
    if lidar_data is None:
        return 0.,0.

    interpreter.set_tensor(input_index, lidar_data)
    start_time = time.time()
    interpreter.invoke()
    # Calculate inference time in milliseconds
    inf_time = (time.time() - start_time) * 1000
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f'output:{output}')
    servo = output[0, 0]  # Extract predicted servo angle
    speed = output[0, 1]  # Extract predicted speed

    return servo, speed

# Linear mapping function


def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

# Undo min-max scaling


def undo_min_max_scale(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


# ROS Publisher-Subscriber
servo_pub = rospy.Publisher(drive, AckermannDriveStamped, queue_size=10)
rospy.Subscriber(lid, LaserScan, callback)
rospy.Subscriber("/vesc/odom", Odometry, odom_callback)


start_ts = time.time()  # Start time
# Load the TensorFlow Lite model
load_model()

# Main loop
while not rospy.is_shutdown():
    # Run the model inference and control the car
    ts = time.time()
    msg = AckermannDriveStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "base_link"

    # Get servo and speed predictions from the model
    servo, speed = dnn_output()

    # Map speed from model's output range to actual speed range
    speed = linear_map(speed, 0, 1, -0.3, 8.0)

    # Print info and write to CSV
    print(f'Servo: {servo}, Speed: {speed}')

    # Assign the speed and steering angle to the message
    msg.drive.speed = speed
    msg.drive.steering_angle = servo

    # Publish the message
    servo_pub.publish(msg)

    # Calculate and print execution time
    dur = time.time() - ts
    if dur > period:
        print("%.3f: took %d ms - deadline miss." %
              (ts - start_ts, int(dur * 1000)))
    else:
        print("%.3f: took %d ms" % (ts - start_ts, int(dur * 1000)))

    rate.sleep()

# End of program
print('\n-----------Recording Completed-----------')
