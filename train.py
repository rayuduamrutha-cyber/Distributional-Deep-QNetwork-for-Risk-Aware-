import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import VehicularEnvironment
from model import DistributionalDQN
from replay_memory import ReplayMemory


EPISODES = 300
STEPS = 30

BATCH_SIZE = 64

GAMMA = 0.95

LEARNING_RATE = 0.001

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


env = VehicularEnvironment()

model = DistributionalDQN()

target_model = DistributionalDQN()

target_model.load_state_dict(
    model.state_dict()
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

memory = ReplayMemory(10000)


def choose_action(state, epsilon):

    if np.random.random() < epsilon:

        return np.random.randint(3)

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model.get_q_values(
            state_tensor
        )

    return q_values.argmax().item()


rewards_history = []


for episode in range(EPISODES):

    state = env.reset()

    total_reward = 0

    for step in range(STEPS):

        action = choose_action(
            state,
            EPSILON
        )

        next_state, reward, done, info = env.step(
            action
        )

        memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        if len(memory) >= BATCH_SIZE:

            batch = memory.sample(
                BATCH_SIZE
            )

            states = torch.FloatTensor(
                np.array(
                    [x[0] for x in batch]
                )
            )

            actions = torch.LongTensor(
                [x[1] for x in batch]
            )

            rewards = torch.FloatTensor(
                [x[2] for x in batch]
            )

            next_states = torch.FloatTensor(
                np.array(
                    [x[3] for x in batch]
                )
            )

            q_values = model.get_q_values(
                states
            )

            current_q = q_values[
                range(BATCH_SIZE),
                actions
            ]

            with torch.no_grad():

                next_q = target_model.get_q_values(
                    next_states
                ).max(1)[0]

                target_q = (
                    rewards
                    + GAMMA * next_q
                )

            loss = nn.MSELoss()(
                current_q,
                target_q
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    EPSILON = max(
        EPSILON_MIN,
        EPSILON * EPSILON_DECAY
    )

    rewards_history.append(
        total_reward
    )

    if episode % 10 == 0:

        target_model.load_state_dict(
            model.state_dict()
        )

        print(
            "Episode:",
            episode,
            "Reward:",
            round(total_reward, 2),
            "Epsilon:",
            round(EPSILON, 3)
        )


torch.save(
    model.state_dict(),
    "trained_model.pth"
)

np.save(
    "rewards.npy",
    rewards_history
)

print("\nTraining completed!")

print(
    "Model saved as trained_model.pth"
)