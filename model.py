import torch
import torch.nn as nn


class DistributionalDQN(nn.Module):

    def __init__(
        self,
        state_size=6,
        action_size=3,
        atoms=51
    ):

        super().__init__()

        self.action_size = action_size

        self.atoms = atoms

        self.v_min = -5

        self.v_max = 5

        self.support = torch.linspace(
            self.v_min,
            self.v_max,
            atoms
        )

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Linear(
                128,
                action_size * atoms
            )

        )

    def forward(self, state):

        output = self.network(state)

        output = output.view(
            -1,
            self.action_size,
            self.atoms
        )

        probabilities = torch.softmax(
            output,
            dim=-1
        )

        return probabilities

    def get_q_values(self, state):

        probabilities = self.forward(state)

        support = self.support.to(
            probabilities.device
        )

        q_values = (
            probabilities * support
        ).sum(dim=2)

        return q_values