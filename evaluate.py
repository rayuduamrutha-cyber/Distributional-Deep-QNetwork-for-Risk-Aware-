import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import VehicularEnvironment
from model import DistributionalDQN


env = VehicularEnvironment()

model = DistributionalDQN()

model.load_state_dict(
    torch.load(
        "trained_model.pth",
        weights_only=True
    )
)

model.eval()


counts = {

    "V2V": 0,

    "Relay": 0,

    "Infrastructure": 0

}


for i in range(300):

    state = env.reset()

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = model.get_q_values(
            state_tensor
        )

    action = q_values.argmax().item()

    if action == 0:

        counts["V2V"] += 1

    elif action == 1:

        counts["Relay"] += 1

    else:

        counts["Infrastructure"] += 1


print(counts)


plt.bar(
    counts.keys(),
    counts.values()
)

plt.xlabel(
    "Communication Mode"
)

plt.ylabel(
    "Number of Selections"
)

plt.title(
    "Mode Selection Distribution"
)

plt.savefig(
    "mode_selection.png",
    dpi=300
)

plt.show()