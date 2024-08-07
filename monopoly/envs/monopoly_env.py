import gymnasium as gym
from gymnasium import spaces
# from gymnasium.spaces import Box
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
class Player:
    def __init__(self, num, ally_num, tag="player1"):
        self.tag = tag
        self.pos = 0
        self.num = num
        self.ally_num = ally_num
        self.prev_pos = []
        self.possession_indices = []

    def change_pos(self, roll_dice, num_states):
        self.pos = (self.pos + roll_dice) % num_states

    def buy(self, board):
        self.possession_indices.append(self.pos)
        board[self.pos] = self.num
        return board

    def sell(self, board, p2_num):
        if self.pos in self.possession_indices:
            self.possession_indices.remove(self.pos)
            board[self.pos] = p2_num
        else:
            pass
        return board


class MonopolyEnv (gym.Env):
    def __init__(self, num_states,num_agents,max_turns=100):
        self.current_player = None
        self.current_player_index = None

        # self.actions = ["skip", "buy", "give"]

        # 0 -skip , 1 - buy ,  2- give
        self.action_space = spaces.Discrete(3)
        self.action = ""
        # self.done = False
        self.episode_length = 0
        # self.no_operation = False

        self.terminated = False
        self.truncated = False

        self.num_states = num_states
        self.num_agents = num_agents
        self.board = np.zeros(self.num_states)

        self.state_observation = np.concatenate(([0], self.board))

        print(np.array(self.state_observation))
        print(np.array(self.state_observation).shape)

        self.observation_space = spaces.Box(
            low=0, high=((self.num_agents+1)**self.num_states), shape=(self.num_states+1,), dtype=np.float64
        )

        self.max_turns = max_turns
        self.players = []
        self.reward = 0

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.episode_length = 0
        self.reward = 0

        # self.x = 0
        self.board = np.zeros(self.num_states)

        self.players = [Player(1, 2, "player1"), Player(2, 1, "player2")]
        # for i in range(num_agents):
        #     players.append(Player(i+1, f"player{i+1}"))
        # players[0] = Player(1, 2, "player1")
        # players[1] = Player(2, 1, "player2")

        # to denote the active player index
        self.current_player_index = 0

        # to denote the active player
        self.current_player = self.players[self.current_player_index]

        # self.state_observation = [self.x, self.board]
        self.state_observation = np.concatenate(([self.current_player.pos], self.board))

        self.observation_space = spaces.Box(
            low=0, high=((self.num_agents + 1) ** self.num_states), shape=(self.num_states+1,), dtype=np.float64
        )

        # return [self.current_player.pos, self.board]
        return self.state_observation, {}

    def action_space(self):
        return self.action_space

    def update_position_roll(self):
        roll = np.random.randint(low=1, high=3)
        self.current_player.change_pos(roll, self.num_states)
        # self.x = (self.x + roll) % self.num_states
        # return [self.x, self.board]
        return [self.current_player.pos, self.observation_space]

    def step(self, action):
        if self.episode_length > self.max_turns:
            self.terminated = True
            self.truncated = True
            return self.state_observation, self.reward, self.terminated, self.truncated,{} #self.episode_length

        if np.all(self.observation_space == 2):
            self.terminated = True
            self.truncated = False
            return self.state_observation, self.reward, self.terminated, self.truncated,{} #self.episode_length

        self.action = action
        # self.action = self.actions[action]
        self.reward = self.get_reward()
        self.state_observation = self.take_action()
        self.episode_length += 1
        # self.no_operation = False

        if (self.episode_length >= self.max_turns):
            self.terminated = True

        return self.state_observation, self.reward, self.terminated, self.truncated,{} # self.episode_length

    def get_reward(self):
        '''
        Return value : rewards
        Input argument.
        '''
        self.reward = 0
        if self.board[self.current_player.pos] == 0:
            if self.action == "buy":
                self.reward += 0
            elif self.action == "give":
                # Invalid action
                self.reward += -10
            else:
                # Skipping even when it can buy
                self.reward += -2

        elif self.board[self.current_player.pos] == 1:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Good action
                self.reward += -1
            else:
                # Skipping even when it can sell
                self.reward += -2

        else:
            if self.action == "buy":
                # Trying to buy already bought land
                self.reward += -1
            elif self.action == "give":
                # Invalid action
                self.reward += -1
            else:
                # Skipping correct action
                self.reward += 0

        if np.all(self.board == 2):
            self.reward += 5
        else:
            self.reward += -1

        return self.reward

    def take_action(self):
        if self.action == "buy":
            if int(self.board[self.current_player.pos]) == 0:
                self.board[self.current_player.pos] = self.current_player.num

        elif self.action == "give":
            if int(self.board[self.current_player.pos]) == self.current_player.num:
                self.board[self.current_player.pos] = self.current_player.ally_num

        return np.concatenate(([self.current_player.pos], self.board))

#
# '''Functions for the agent'''
#
#
# def current_state_to_string(state):
#     x = state[0]
#     board = state[1]
#     # current_state = ''.join(x).join(board[x])
#     current_state = f"{int(x)}{int(board[x])}"
#
#     # current_state = ''.join(str(int(e)) for e in state)
#     return current_state
#
#
# # def current_state_to_string(state):
# #     current_state = ''.join(str(int(e)) for e in state)
# #     return current_state
# # self.x self.board
# def best_state_action_value(current_state):
#     max_val = np.inf * -1
#     #     from IPython.core.debugger import Tracer; Tracer()()
#     for key in current_state.keys():
#         if current_state[key] > max_val:
#             max_val = current_state[key]
#             best_action = key
#     return best_action, max_val
#
#
# def get_all_states_as_strings(num_states, num_agents):
#     states = []
#     for i in range(num_states):
#         for j in range(num_agents + 1):
#             tmp = [i, j]
#             states.append("".join(str(a) for a in tmp))
#     # print(states)
#     return states
#
#
# def initialize_Q(num_states, num_agents):
#     Q = {}
#     states = get_all_states_as_strings(num_states, num_agents)
#     for state in states:
#         Q[state] = {}
#         for i in range(3):  # Number of actions = 3
#             Q[state][i] = np.random.uniform(-2, 2, 1)
#             # Q[state][i] = -10000
#     return Q
#
#
# if __name__ == "__main__":
#     # MAX_STATES = MAX_HOR_LENGTH*MAX_HOR_LENGTH
#     num_states = 4
#     num_agents = 2
#
#     TOTAL_EPISODES = 1000
#     SIM_RUN = 10000
#     SHOW_EVERY = 10000
#     OBSERVATION_SPACE = 2
#     # LEARNING_RATE = 0.05 # alpha in the literature
#     LEARNING_RATE = 0.05  # alpha in the literature
#     DISCOUNT = 0.95  # gamma IN the literature
#     EPSILON = 0.1
#     START_EPSILON_DECAYING = 150
#     END_EPSILON_DECAYING = 600
#     epsilon_decay_value = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
#     debug = False
#     Summed_reward = []
#
#     for sim in range(SIM_RUN):
#         EPSILON = 0.1
#         done = False
#
#         agent = MonopolyEnv(num_states, 40)
#         Q_table = initialize_Q(num_states, num_agents)
#
#         Total_running_reward = []
#         action_space = agent.action_space()
#         action_indexes = [i for i in range(len(action_space))]
#         cnt = 0
#         no_op = False
#         episode_lengths = []
#         for episode in range(TOTAL_EPISODES):
#             done = False
#             current_state = agent.reset()
#             cnt += 1
#             total_episode_reward = 0
#             episode_length = 0
#
#             while not done:
#                 current_state = agent.update_position_roll()
#                 if debug:
#                     print(f"currentagent: {agent.current_player.num}, Agent.x: {current_state[0]}")
#                     if agent.current_player.num != 1:
#                         print("SOMETHING IS WRONG!!!")
#
#                 current_state_str = current_state_to_string(current_state)
#                 kind_of_selection_ = 'None'
#
#                 if np.random.uniform() > EPSILON:
#                     action, max_qt1 = best_state_action_value(Q_table[current_state_str])
#                     kind_of_selection_ = 'Greedy'
#                 else:
#                     action = np.random.choice(action_indexes)
#                     max_qt1 = Q_table[current_state_str][action]
#                     kind_of_selection_ = 'Random'
#
#                 if debug:
#                     print(f"action: {action_space[action]}, current_state: {current_state_str}")
#                     print(f"Before action: {current_state}")
#
#                 next_state, reward, done, no_op, episode_length = agent.step(action_space[action])
#                 total_episode_reward += reward
#
#                 if debug:
#                     print(f"After action: {next_state}")
#                     print(f"reward: {reward}")
#                     print(f"---------------------------------------")
#
#                 Q_table[current_state_str][action] += LEARNING_RATE * (
#                             reward + DISCOUNT * max_qt1 - Q_table[current_state_str][action])
#                 #             print(f'current state : {current_state}. Action : {action_space[action]}. Next state: {next_state}. Kind of Sel: {kind_of_selection_}')
#                 current_state = next_state
#                 # print(current_state)
#                 cnt += 1
#
#             Total_running_reward.append(total_episode_reward)
#             episode_lengths.append(episode_length)
#
#             if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
#                 EPSILON -= epsilon_decay_value
#
#             # print(f"---------------------------------------")
#
#         if sim == 0:
#             Summed_reward = Total_running_reward
#         else:
#             Summed_reward = np.vstack((Summed_reward, Total_running_reward))
#
#         if sim % SHOW_EVERY == 0:
#             print(sim)
#
#     # Displaying average reward
#
#     df = pd.DataFrame(Summed_reward)
#     Mean_total_reward = df.mean()
#     Mean_total_reward
#     # print('--------------------------')
#     # print(f'Numer of steps per episode : {episode_length}. Reward : {Total_running_reward}')
#
#     plt.plot(Mean_total_reward)
#     plt.grid()
#     plt.title(f'Mean reward after {SIM_RUN} simulation of {TOTAL_EPISODES} Episode each')
#     plt.xlabel('Episodes')
#     plt.ylabel('Rewards / Costs')
#     plt.show()
#
#     # Moving average
#     N = 100
#     cumsum, moving_aves = [0], []
#
#     for i, x in enumerate(Total_running_reward, 1):
#         cumsum.append(cumsum[i - 1] + x)
#         if i >= N:
#             moving_ave = (cumsum[i] - cumsum[i - N]) / N
#             # can do stuff with moving_ave here
#             moving_aves.append(moving_ave)
#
#     plt.plot(moving_aves)
#     plt.title('Moving Average')
#     plt.show()
#
#     plt.plot(episode_lengths)