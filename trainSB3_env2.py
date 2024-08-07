from stable_baselines3 import PPO, A2C, DQN
import os
from monopoly.envs.monopoly_env2 import MonopolyEnv2
import time

models_dir = f"models/{int(time.time())}/1000_plotdiff/"
logdir = f"logs/{int(time.time())}/1000_plotdiff/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = MonopolyEnv2(12, 6, 2, 1000)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1
iters = 0
position_list = [0,0]
file_path = "ownership_data.txt"
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
    num_ep = 0
    print("ENV EP LEN: ", env.episode_length)
    if env.episode_length<=70 and num_ep<10:
        print(f"Current_player: {env.current_player.num}")
        print(f"Position before roll: {env.current_player.pos}")
        random_action = env.action_space.sample()
        print("action", random_action)

        obs, reward, done, trunc, info = env.step(random_action)
        print(f"Roll: {env.roll_val}")
        print(f"Position after roll: {env.current_pos}")
        print('state', [format(num, '.2f') for num in obs])
        print('reward', reward)
        print(info)
        owner = []
        worths = []
        for city in env.board:
            if city.owner != None:
                owner_num = city.owner.num
            else:
                owner_num = 0
            owner.append([city.name, owner_num])
        for player in env.players:
            worths.append([player.num, player.money])
        print(owner)
        print(worths)
        print()
        owner_tuple = [(location, value) for location, value in owner]
        position_list[env.current_player.num-1] = env.current_pos
        with open(file_path, 'a') as file:
            file.write(str(owner_tuple)+"\n")
            file.write(str(env.current_player.num)+"\n")
            file.write(str(env.roll_val)+"\n")
            file.write(str(env.actions[random_action])+"\n")
            file.write(str(position_list)+"\n")
        num_ep += 1

###### RUN THE BELOW CODE IN CMD FOR PROGRESS GRAPHS #######
# tensorboard --logdir=logs
