from stable_baselines3 import PPO #PPO
import os
from environment import CarEnv
import time
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np




class MetricsLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        
        if hasattr(env, 'reward_log'):
            mean_reward = np.mean(env.reward_log)
            mean_distance = np.mean(env.distance_travelled_log)
            mean_steering = np.mean(env.steering_log)
            mean_speed = np.mean(env.speed_log)

            collision_count = len(env.collision_hist)
            lane_invade_count = len(env.lane_invade_hist)

            # Log metrics to TensorBoard
            self.logger.record('reward/mean_reward', mean_reward)
            self.logger.record('metrics/mean_distance', mean_distance)
            self.logger.record('metrics/mean_steering', mean_steering)
            self.logger.record('metrics/mean_speed', mean_speed)
            self.logger.record('events/collision_count', collision_count)
            self.logger.record('events/lane_invade_count', lane_invade_count)

            # Dump logs to TensorBoard every step
            self.logger.dump(self.num_timesteps)

        return True


print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir, name_prefix='model')

env = CarEnv()

env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)

TIMESTEPS = 100000 # how long is each training iteration - individual steps
iters = 0
while iters<4:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=[MetricsLogger(), checkpoint_callback])
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")