import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


def environment(**kwargs):
    lake = np.zeros(4 * 4).reshape(4, 4)
    env_dimesion = [(x, y) for x in [0, 1, 2, 3] for y in [0, 1, 2, 3]]
    for kind, point in kwargs.items():
        if "punish" in kind:
            lake[env_dimesion[point]] = -1
        if "reward" in kind:
            lake[env_dimesion[point]] = 1
        if "start" in kind:
            start = env_dimesion[point]
    return lake, start


def moving_direction(current_state, moving_direction):
    """
    args :
        current_state : array like this is for chosen from action_map
        moving_direction : "up" , "left" , "right" , "down"
    ouput:
        selected state for environment
    """

    edge_left, edge_right, k = (
        [4 * x for x in range(0, 4)],
        [4 * k - 1 for k in range(1, 5)],
        0,
    )

    for j, i in enumerate(current_state):
        if j == 0:
            k = 4 * i
        else:
            k += i

    current_state = k
    if moving_direction == "up":
        new_state = (
            lambda current_state: current_state - 4
            if current_state >= 4
            else "don't move"
        )
    elif moving_direction == "down":
        new_state = (
            lambda current_state: current_state + 4
            if current_state <= 11
            else "don't move"
        )
    elif moving_direction == "right":
        new_state = (
            lambda current_state: current_state + 1
            if current_state <= 14
            and current_state >= 0
            and not current_state in edge_right
            else "don't move"
        )
    elif moving_direction == "left":
        new_state = (
            lambda current_state: current_state - 1
            if (current_state <= 15)
            and (current_state >= 1)
            and (not current_state in edge_left)
            else "don't move"
        )
    if new_state(current_state) != "don't move":
        first = int(new_state(current_state) / 4)
        second = new_state(current_state) % 4
        return [first, second]
    else:
        return new_state(current_state)


def actions_map(lake):
    """
    get environment and return rewards, actions, end whcih corresponding to this environment
    """
    end = {}
    rewards = {}
    actions = {
        (0, 0): ["right", "down"],
        (0, 1): ["right", "left", "down"],
        (0, 2): ["right", "left", "down"],
        (0, 3): ["down", "left"],
        (1, 0): ["up", "down", "right"],
        (1, 1): ["up", "down", "right", "left"],
        (1, 2): ["up", "down", "right", "left"],
        (1, 3): ["up", "down", "left"],
        (2, 0): ["up", "right", "down"],
        (2, 1): ["left", "right", "down", "up"],
        (2, 2): ["up", "down", "left", "right"],
        (2, 3): ["up", "down", "left"],
        (3, 0): ["up", "right"],
        (3, 1): ["up", "right", "left"],
        (3, 2): ["up", "right", "left"],
        (3, 3): ["left", "up"],
    }

    for x in range(0, len(lake)):
        for y in range(0, len(lake)):
            if lake[x, y] == -1 or lake[x, y] == 1:
                end_dict = {(x, y): lake[x, y]}
                end.update(end_dict)
                rewards.update(end_dict)
                del actions[(x, y)]
            else:
                rewards.update({(x, y): 0})

    return rewards, actions, end


def exploit(state, q_table, lake):
    """
    get current state and q_table and return maximum value around current state
    """
    rewards, action, end = actions_map(lake)
    next_state = 0
    q_table = lake
    current_reward = q_table[tuple(state)]
    lengh, width = lake.shape
    done = False
    x, y = state[0], state[1]
    up, down, left, right = [x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]
    if x >= 1:
        reward_up = q_table[tuple(up)]
    else:
        reward_up = "wall"
    if (lengh - 1) - x >= 1:
        reward_down = q_table[tuple(down)]
    else:
        reward_down = "wall"
    if (width - 1) - y >= 1:
        reward_right = q_table[tuple(right)]
    else:
        reward_right = "wall"
    if y >= 1:
        reward_left = q_table[tuple(left)]
    else:
        reward_left = "wall"

    list_move = [reward_up, reward_down, reward_left, reward_right]
    list_integer = []
    for i in list_move:
        if i != "wall":
            list_integer.append(i)
    max_reward = max(list_integer)

    if max_reward == reward_up:
        next_state = up
    if max_reward == reward_down:
        next_state = down
    if max_reward == reward_left:
        next_state = left
    if max_reward == reward_right:
        next_state = right
    if tuple(next_state) in end.keys():
        done = True
        max_reward = end[tuple(next_state)]

    return next_state, max_reward, current_reward, done


def explore(state, q_table, lake):
    """
    this is our engine function
    """
    rewards, actions, end = actions_map(lake)
    current_reward = q_table[tuple(state)]
    done = False

    next_action = actions[tuple(state)]
    k = np.random.randint(0, len(next_action))
    our_action = next_action[k]
    next_state = moving_direction(state, our_action)

    if (
        tuple(next_state) in end.keys()
    ):  ## specify it is it rewards or punish, this is finish --> done = True
        done = True
        next_value = end[tuple(next_state)]
    else:
        next_value = q_table[tuple(next_state)]

    return next_state, next_value, current_reward, done


class Agent:
    def __init__(self, env, start_point, lr, gamma):
        self.env = env
        self.initial_state = start_point  ## start
        self.length, self.width = self.env.shape
        self.learning_rate = lr
        self.discount_rate = gamma
        self.max_exploration_rate = 0.8
        self.min_exploration_rate = 0.05
        self.exploration_decay_rate = 0.01
        self.exploration_rate = 0.8
        self.explore_list = []

    def search(self):
        number_episode = 500
        max_steps_per_episode = 2000
        table = np.zeros((self.env.shape))

        for episode in range(number_episode):
            position = self.initial_state
            rewards_current_episode = 0
            done = False

            for _ in range(max_steps_per_episode):
                exploration_rate_threshold = random.uniform(0, 1)

                if exploration_rate_threshold > self.exploration_rate:  ## exploitation
                    next_state, next_value, current_reward, done = exploit(
                        position, table, self.env
                    )
                else:
                    next_state, next_value, current_reward, done = explore(
                        position, table, self.env
                    )
                ## update q_table:
                table[tuple(position)] = table[tuple(position)] * (
                    1 - self.learning_rate
                ) + self.learning_rate * (
                    current_reward + self.discount_rate * next_value
                )
                position = next_state
                rewards_current_episode += current_reward

                if done == True:
                    break

            if episode % 20 == 0:
                print(f"----------------> episode number {episode} updated Q_Table")
                print("-----------------> updated Q_table \n  ", table)
            self.exploration_rate = self.min_exploration_rate + (
                self.max_exploration_rate - self.min_exploration_rate
            ) * np.exp(-self.exploration_decay_rate * episode)
            self.explore_list.append(self.exploration_rate)

        return table, self.explore_list


def show_result(result_table, lake, exploration_rate, end):

    for pointer, point in end.items():
        result_table[pointer] = point

    print("--------------------------------------------------------")
    print(" our initial enivronment is \n\n", lake)
    print(" our updated q_table is     \n\n", result_table)

    plt.plot(np.array(exploration_rate), label="exploration_reate_decay")
    plt.legend()
    plt.show()

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)
    ax0 = sns.heatmap(lake, cmap="rocket_r", annot=True, ax=ax0, linewidths=0.7)
    ax0.set_title(" before agent start actions ")

    ax1 = sns.heatmap(result_table, cmap="rocket_r", annot=True, ax=ax1, linewidths=0.7)
    ax1.set_title(" after agent discover environment ")
    plt.tight_layout()
    plt.show()


def main():
    lake, start = environment(reward=12, reward_1=3, punish_2=15, start=5)
    end = actions_map(lake)[2]
    lizard = Agent(lake, start_point=list(start), lr=0.2, gamma=0.8)
    final_table, exploration_decay = lizard.search()
    show_result(
        result_table=final_table, lake=lake, end=end, exploration_rate=exploration_decay
    )


if __name__ == "__main__":
    main()
