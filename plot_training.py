import numpy as np
import matplotlib.pyplot as plt


rewards = np.load(
    "rewards.npy"
)


plt.figure(figsize=(10, 5))

plt.plot(
    rewards
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Training Reward"
)

plt.title(
    "Distributional DQN Training Reward"
)

plt.grid()

plt.show()