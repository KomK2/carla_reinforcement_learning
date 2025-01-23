import os
import time
import cv2
import gym
import numpy as np
import carla
from stable_baselines3 import PPO
from environment import CarEnv  # Update import if your environment file is named `environment.py`

# Define the model path
MODEL_PATH = '/home/kiran/models/1733966910/400000.zip'

# Load the trained model
print(f"Loading model from {MODEL_PATH}...")
model = PPO.load(MODEL_PATH)

# Create the CARLA environment
env = CarEnv()

# Reset the environment
obs = env.reset()

# Set up inference loop variables
done = False
total_reward = 0
steps = 0
max_steps = 5000  # Adjust as needed

# Run the inference loop
while not done and steps < max_steps:
    steps += 1
    
    # Predict the action
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action in the environment
    obs, reward, done, info = env.step(action)
    
    # Accumulate reward
    total_reward += reward
    
    # Print status every 10 steps
    if steps % 10 == 0:
        print(f"Step: {steps}, Reward: {reward}, Total Reward: {total_reward}")
    
    # Display camera feed if enabled
    # if env.SHOW_CAM and env.front_camera is not None:
    #     # Ensure the image is in the correct format for OpenCV
    #     camera_feed = cv2.cvtColor(env.front_camera, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    #     cv2.imshow("Carla Camera", camera_feed)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
    #         break

# Cleanup the environment
env.cleanup()
cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed
print(f"Inference completed. Total Reward: {total_reward}")
