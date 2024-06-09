import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from src.map_config import BLOCK_COLOR, BLOCK_ID

MAP_FOLDER = "map"


class MapCreatorApp:
    def __init__(self, root, n, map_folder):
        self.root = root
        self.root.title("Map Creator App")

        # Attributes
        self.n = n
        self.map_folder = map_folder

        # Initialize the map
        self.map = np.zeros((self.n, self.n), dtype=int)
        self.cell_size = 30

        # Track the positions of start and end blocks
        self.start_pos = None
        self.end_positions = []  # Modified to store multiple end positions

        # Create the canvas
        self.canvas = tk.Canvas(
            self.root, width=self.n * self.cell_size, height=self.n * self.cell_size
        )
        self.canvas.grid(row=1, column=0, columnspan=5)

        # Create the paint buttons
        self.paint_color = BLOCK_ID["open_field"]  # Default paint color (open field)
        self.create_buttons()

        # Draw the initial grid
        self.draw_grid()

        # Bind mouse click and motion events for painting
        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint_cell)

    def create_buttons(self):
        labels = list(BLOCK_COLOR.keys())
        colors = list(BLOCK_COLOR.values())
        self.buttons = []

        for color, label in zip(colors, labels):
            i = BLOCK_ID[label]
            button = tk.Button(
                self.root,
                text=label,
                bg=color,
                width=10,
                height=2,
                command=lambda c=i: self.set_paint_color(c),
            )
            button.grid(row=0, column=i)
            self.buttons.append(button)

        # Finish button
        finish_button = tk.Button(
            self.root, text="Finish", width=50, height=2, command=self.finish
        )
        finish_button.grid(row=2, column=0, columnspan=5)

    def set_paint_color(self, color_index):
        self.paint_color = color_index

    def draw_grid(self):
        for i in range(self.n):
            for j in range(self.n):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="black", fill=BLOCK_COLOR["open_field"]
                )

    def paint_specific_cell(self, row, col, color):
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color)

    def start_paint(self, event):
        self.paint_cell(event)

    def paint_cell(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        if x < self.n and y < self.n:
            current_block = self.map[y, x]

            # Ensure only one start point
            if self.paint_color == BLOCK_ID["start"]:  # Start point
                if self.start_pos and self.start_pos != (y, x):
                    old_y, old_x = self.start_pos
                    self.map[old_y, old_x] = BLOCK_ID[
                        "open_field"
                    ]  # Reset old start to open_field
                    self.paint_specific_cell(old_y, old_x, BLOCK_COLOR["open_field"])
                self.start_pos = (y, x)
            elif self.paint_color == BLOCK_ID["end"]:  # End point
                if (y, x) not in self.end_positions:
                    self.end_positions.append((y, x))
            elif current_block == BLOCK_ID["start"]:  # Overwriting a start point
                self.start_pos = None
            elif current_block == BLOCK_ID["end"]:  # Overwriting an end point
                if (y, x) in self.end_positions:
                    self.end_positions.remove((y, x))

            color = list(BLOCK_COLOR.values())[self.paint_color]
            self.map[y, x] = self.paint_color
            self.paint_specific_cell(y, x, color)

    def finish(self):
        if not self.start_pos or not self.end_positions:
            messagebox.showerror(
                "Error", "Please place both a start point and at least one end point."
            )
            return
        if not os.path.exists(MAP_FOLDER):
            os.makedirs(MAP_FOLDER)
        filename = simpledialog.askstring("Save map", "Enter the filename to save:")
        if filename:
            file_path = f"{self.map_folder}/{filename}.npy"
            np.save(file_path, self.map)
        self.root.quit()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()

    # Ask user for map size
    map_size = simpledialog.askinteger(
        "Input", "Enter the size of the map (e.g., 15):", minvalue=5, maxvalue=26
    )
    if map_size is None:
        root.quit()  # Exit if no size is provided

    app = MapCreatorApp(root, map_size, MAP_FOLDER)
    root.mainloop()
