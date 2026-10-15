import torch
import numpy as np

from environment import VehicularEnvironment
from model import DistributionalDQN


# --------------------------------
# Load environment
# --------------------------------

env = VehicularEnvironment()


# --------------------------------
# Load trained model
# --------------------------------

model = DistributionalDQN()

model.load_state_dict(
    torch.load(
        "trained_model.pth",
        weights_only=True
    )
)

model.eval()


mode_names = [
    "V2V",
    "Relay",
    "Infrastructure"
]


print("\n")
print("=" * 60)
print("     RISK-AWARE VEHICULAR MODE SELECTION")
print("=" * 60)


for simulation in range(10):

    # Generate network condition

    state = env.reset()


    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)


    # AI prediction

    with torch.no_grad():

        q_values = model.get_q_values(
            state_tensor
        )


    action = q_values.argmax().item()


    # Get communication result

    _, reward, _, info = env.step(
        action
    )


    print("\n")
    print("-" * 60)

    print(
        "SIMULATION:",
        simulation + 1
    )

    print("-" * 60)


    print(
        "Distance       :",
        round(info["distance"], 2),
        "m"
    )


    print(
        "Vehicle Speed  :",
        round(info["speed"], 2),
        "km/h"
    )


    print(
        "Signal Quality :",
        round(info["signal"], 3)
    )


    print(
        "Packet Loss    :",
        round(info["packet_loss"], 3)
    )


    print(
        "Latency        :",
        round(info["latency"], 2),
        "ms"
    )


    print(
        "Network Load   :",
        round(info["network_load"], 3)
    )


    print("\nAI DECISION")

    print(
        "V2V Q-value            :",
        round(q_values[0][0].item(), 3)
    )


    print(
        "Relay Q-value          :",
        round(q_values[0][1].item(), 3)
    )


    print(
        "Infrastructure Q-value:",
        round(q_values[0][2].item(), 3)
    )


    print(
        "\nSELECTED MODE:",
        mode_names[action]
    )


    print(
        "Reward:",
        round(reward, 3)
    )


print("\n")
print("=" * 60)
print("              DEMO COMPLETED")
print("=" * 60)