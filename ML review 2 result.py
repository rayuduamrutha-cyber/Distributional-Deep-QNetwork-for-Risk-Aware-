import numpy as np

ACTION_NAMES = [
    "V2V",
    "Relay",
    "Infrastructure"
]


class VehicularEnvironment:

    def __init__(self):

        self.state_size = 6
        self.action_size = 3

        self.state = None

    def reset(self):

        self.state = np.random.rand(self.state_size)

        return self.state

    def step(self, action):

        distance = 10 + self.state[0] * 490
        speed = 10 + self.state[1] * 90
        signal = self.state[2]
        packet_loss = self.state[3]
        latency = 10 + self.state[4] * 190
        network_load = self.state[5]

        if action == 0:

            mode = "V2V"

            quality = (
                1.2 * signal
                + 0.8 * (1 - distance / 500)
                - 0.8 * network_load
            )

            mode_latency = (
                latency * 0.75
                + distance * 0.025
            )

            mode_packet_loss = min(
                1,
                packet_loss * 0.85
                + distance / 900
            )

        elif action == 1:

            mode = "Relay"

            quality = (
                signal
                + 0.7 * (1 - distance / 500)
                - 0.55 * network_load
            )

            mode_latency = (
                latency * 0.95
                + distance * 0.015
            )

            mode_packet_loss = min(
                1,
                packet_loss * 0.65
                + distance / 1100
            )

        else:

            mode = "Infrastructure"

            quality = (
                0.95 * signal
                + 0.35 * (1 - distance / 500)
                - 0.25 * network_load
            )

            mode_latency = (
                latency * 0.65
                + distance * 0.006
            )

            mode_packet_loss = min(
                1,
                packet_loss * 0.45
            )

        reliability = 1 - mode_packet_loss

        reward = (
            quality
            + reliability
            - 0.01 * mode_latency
        )

        reward += np.random.normal(0, 0.1)

        new_state = np.random.rand(self.state_size)

        self.state = new_state

        info = {

            "mode": mode,

            "distance": distance,

            "speed": speed,

            "signal": signal,

            "packet_loss": mode_packet_loss,

            "latency": mode_latency,

            "network_load": network_load,

            "reliability": reliability

        }

        return new_state, reward, False, info