import numpy as np
from src.map_config import BLOCK_PENALIZATION, BLOCK_ID


class Environment:
    """
    The Environment class represents the island environment for the dynamic \
    programming solution.

    Attributes:
        map (np.ndarray): The map of the island.
        map_size (int): The size of the map (number of rows/columns).
        current_state (np.ndarray): The current state of the environment.
        final_state (np.ndarray): The final state of the environment.
        S (np.ndarray): An array of all possible states.
        S_coords (np.ndarray): An array of coordinates for each state.
        initial_s (int): The initial state index.
        S_terminal (np.ndarray): An array of terminal state indices.
        A (list): A list of all possible actions.
    """

    ACTIONS = {
        "up": 0,
        "down": 1,
        "right": 2,
        "left": 3,
    }

    def __init__(self, map: np.ndarray):
        self.map = map
        self.map_size = len(map)
        self.current_state = np.array((0, 0))
        self.final_state = np.array((len(map) - 1, len(map) - 1))
        self.S = np.array([i for i in range(len(map) ** 2)])
        self.S_coords = np.array(
            [[i, j] for j in range(len(map)) for i in range(len(map))]
        )  # Note that len(S) = len(S_coords)
        self.initial_s = self._get_initial_state()
        self.S_terminal = self._get_terminal_states()
        self.A = list(__class__.ACTIONS.values())

    def _get_initial_state(self):
        """
        Identifies the initial state in the environment.

        Returns:
            int: The index of the initial state.
        """
        for s in self.S:
            coords = self.S_coords[s]
            state_value = self.map[coords[0]][coords[1]]
            if state_value == BLOCK_ID["start"]:
                return s

    def _get_terminal_states(self):
        """
        Identifies all terminal states in the environment.

        Returns:
            np.ndarray: An array of terminal state indices.
        """
        terminal_states = []
        for s in self.S:
            coords = self.S_coords[s]
            state_value = self.map[coords[0]][coords[1]]
            if state_value == BLOCK_ID["end"]:
                terminal_states.append(s)
        return np.array(terminal_states)

    def get_state_from_coords(self, state_coords: np.ndarray):
        """
        Converts coordinates to the corresponding state index.

        Parameters:
            state_coords (np.ndarray): The coordinates of the state.

        Returns:
            int: The index of the state.
        """
        return state_coords[0] + state_coords[1] * self.map_size

    def step(self, state: int, action: int):
        """
        Applies an action to a given state and returns the resulting next \
        state and reward.

        Parameters:
            state (int): The current state.
            action (int): The action to be applied.

        Returns:
            tuple: The next state and the reward.
        """
        assert action in self.A

        # If state is terminal, stay on the same state
        if state in self.S_terminal:
            return state, BLOCK_PENALIZATION[BLOCK_ID["end"]]

        # Compute next state
        state_coords = self.S_coords[state]
        if action == self.ACTIONS["up"]:
            if state_coords[0] == 0:
                next_state_coords = state_coords
            else:
                next_state_coords = state_coords + np.array([-1, 0])

        if action == self.ACTIONS["down"]:
            if state_coords[0] == self.map_size - 1:
                next_state_coords = state_coords
            else:
                next_state_coords = state_coords + np.array([1, 0])

        if action == self.ACTIONS["right"]:
            if state_coords[1] == self.map_size - 1:
                next_state_coords = state_coords
            else:
                next_state_coords = state_coords + np.array([0, 1])

        if action == self.ACTIONS["left"]:
            if state_coords[1] == 0:
                next_state_coords = state_coords
            else:
                next_state_coords = state_coords + np.array([0, -1])
        next_state = self.get_state_from_coords(next_state_coords)

        # Compute reward
        next_state_value = self.map[next_state_coords[0]][next_state_coords[1]]
        reward = BLOCK_PENALIZATION[next_state_value]

        return next_state, reward
