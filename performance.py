import numpy as np
import matplotlib.pyplot as plt

from environment import VehicularEnvironment


env = VehicularEnvironment()


mode_names = [
    "V2V",
    "Relay",
    "Infrastructure"
]


results = {
    "V2V": [],
    "Relay": [],
    "Infrastructure": []
}


# Run 300 simulations
for i in range(300):

    state = env.reset()

    for action in range(3):

        # Put the same state back into the environment
        env.state = state.copy()

        next_state, reward, done, info = env.step(action)

        results[mode_names[action]].append({

            "reward": reward,

            "latency": info["latency"],

            "packet_loss": info["packet_loss"],

            "reliability": info["reliability"]

        })


# Calculate averages
average_reward = []
average_latency = []
average_packet_loss = []
average_reliability = []


for mode in mode_names:

    average_reward.append(
        np.mean([
            x["reward"]
            for x in results[mode]
        ])
    )

    average_latency.append(
        np.mean([
            x["latency"]
            for x in results[mode]
        ])
    )

    average_packet_loss.append(
        np.mean([
            x["packet_loss"]
            for x in results[mode]
        ])
    )

    average_reliability.append(
        np.mean([
            x["reliability"]
            for x in results[mode]
        ])
    )


print("\n========== PERFORMANCE COMPARISON ==========")


for i, mode in enumerate(mode_names):

    print("\n", mode)

    print(
        "Average Reward:",
        round(average_reward[i], 3)
    )

    print(
        "Average Latency:",
        round(average_latency[i], 3),
        "ms"
    )

    print(
        "Average Packet Loss:",
        round(average_packet_loss[i], 3)
    )

    print(
        "Average Reliability:",
        round(average_reliability[i], 3)
    )


# -------------------------
# Plot 1 - Reward
# -------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    mode_names,
    average_reward
)

plt.xlabel("Communication Mode")

plt.ylabel("Average Reward")

plt.title(
    "Average Reward Comparison"
)

plt.grid(axis="y")

plt.savefig(
    "reward_comparison.png",
    dpi=300
)

plt.show()


# -------------------------
# Plot 2 - Latency
# -------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    mode_names,
    average_latency
)

plt.xlabel("Communication Mode")

plt.ylabel("Latency (ms)")

plt.title(
    "Average Latency Comparison"
)

plt.grid(axis="y")

plt.savefig(
    "latency_comparison.png",
    dpi=300
)

plt.show()


# -------------------------
# Plot 3 - Packet Loss
# -------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    mode_names,
    average_packet_loss
)

plt.xlabel("Communication Mode")

plt.ylabel("Packet Loss")

plt.title(
    "Average Packet Loss Comparison"
)

plt.grid(axis="y")

plt.savefig(
    "packet_loss_comparison.png",
    dpi=300
)

plt.show()


# -------------------------
# Plot 4 - Reliability
# -------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    mode_names,
    average_reliability
)

plt.xlabel("Communication Mode")

plt.ylabel("Reliability")

plt.title(
    "Average Reliability Comparison"
)

plt.grid(axis="y")

plt.savefig(
    "reliability_comparison.png",
    dpi=300
)

plt.show()


print("\nAll comparison graphs created successfully!")