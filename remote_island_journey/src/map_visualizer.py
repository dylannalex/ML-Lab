import numpy as np
import matplotlib.pyplot as plt
from src.environment import Environment
from src.map_config import BLOCK_COLOR, BLOCK_ID


class Map:
    """
    The Map class represents the island map and provides methods for visualizing the map, \
    actions, value functions, and paths.
    """

    EDGE_COLOR = "#000000"
    ARROW_COLOR = "#000000"
    PATH_COLOR = "#292929"

    def __init__(self, env: Environment):
        self._map_env = env

    def visualize_map(self):
        """
        Visualizes the map with blocks.
        """
        fig, ax = plt.subplots()

        self._add_map_blocks_to_axis(ax)
        ax.set_xlim(0, self._map_env.map_size)
        ax.set_ylim(0, self._map_env.map_size)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        plt.show()

    def visualize_actions(self, actions: np.ndarray):
        """
        Visualizes the map with actions represented by arrows.

        Parameters:
            actions (np.ndarray): An array of actions taken by the agent at each state.
        """
        # Create a plot and add map blocks
        fig, ax = plt.subplots()
        self._add_map_blocks_to_axis(ax)

        # Plot actions
        n = self._map_env.map_size
        for state in self._map_env.S:
            if state in self._map_env.S_terminal:
                continue
            state_coords = self._map_env.S_coords[state]
            x, y = state_coords[1], n - 1 - state_coords[0]
            # Draw policy action
            action = actions[state]
            arrow = self._action_to_arrow(action)
            ax.text(
                x + 0.5,
                y + 0.5,
                arrow,
                ha="center",
                va="center",
                color=__class__.ARROW_COLOR,
            )

        # Set plot limits and aspect
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Show the plot
        plt.show()

    def visualize_value_function(self, V: np.ndarray):
        """
        Visualizes the given value function using a color map.

        Parameters:
            V (np.ndarray): The value function to be visualized.
        """
        # Create a plot
        fig, ax = plt.subplots()

        # Draw the map
        n = self._map_env.map_size
        value_matrix = np.zeros((n, n))
        for state in self._map_env.S:
            state_coords = self._map_env.S_coords[state]
            value_matrix[n - state_coords[0] - 1, state_coords[1]] = V[state]

        cax = ax.matshow(value_matrix, cmap="viridis")

        # Add color bar
        cbar = fig.colorbar(cax)
        cbar.set_label("V Intensity")

        # Set plot limits and aspect
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )

        # Show the plot
        plt.show()

    def visualize_path(self, actions):
        """
        Visualizes the path taken by the agent according to the provided actions.

        Parameters:
            actions (np.ndarray): An array of actions taken by the agent at each state.
        """
        # Create a plot and add map blocks
        fig, ax = plt.subplots()
        self._add_map_blocks_to_axis(ax)

        # Find path followed guided by policy
        current_state = self._map_env.initial_s
        path = [current_state]
        while True:
            next_state, _ = self._map_env.step(current_state, actions[current_state])
            path.append(next_state)
            if next_state in path[:-1]:
                # Loop in path
                break
            if next_state in self._map_env.S_terminal:
                # Reached terminal state
                break
            current_state = next_state

        # Plot the path
        n = self._map_env.map_size
        path_coords = [
            (
                self._map_env.S_coords[state][1] + 0.5,
                n - self._map_env.S_coords[state][0] - 0.5,
            )
            for state in path
        ]
        path_x, path_y = zip(*path_coords)
        ax.plot(
            path_x,
            path_y,
            marker="None",
            color=__class__.PATH_COLOR,
            linewidth=3,
            markersize=5,
        )

        # Set plot limits and aspect
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Show the plot
        plt.show()

    def _add_map_blocks_to_axis(self, ax):
        # Draw the map
        n = self._map_env.map_size
        for i in range(n):
            for j in range(n):
                block_id = self._map_env.map[i, j]
                block_name = [
                    key for key, value in BLOCK_ID.items() if value == block_id
                ][0]
                color = BLOCK_COLOR[block_name]
                rect = plt.Rectangle(
                    (j, n - 1 - i),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor=__class__.EDGE_COLOR,
                    facecolor=color,
                )
                ax.add_patch(rect)

    def _action_to_arrow(self, action):
        """
        Converts an action to an arrow symbol for plotting.
        """
        if action == Environment.ACTIONS["up"]:
            return "↑"
        elif action == Environment.ACTIONS["down"]:
            return "↓"
        elif action == Environment.ACTIONS["right"]:
            return "→"
        elif action == Environment.ACTIONS["left"]:
            return "←"
