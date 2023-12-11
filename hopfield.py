# SFC Project: Hopfield Network Image Autoassociation
# Author: Martin Zmitko (xzmitk01)
# Date: 2023-11-27

import os
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

input_data = np.zeros(64 * 64)
cancel_run = False

def read_bw_png(filepath):
    image = Image.open(filepath).convert("L")
    pixels = list(image.getdata())
    binary_pixels = [1 if pixel == 255 else -1 for pixel in pixels]
    return np.array(binary_pixels)

def compute_weight_matrix(filepaths):
    num_pixels = None
    weight_matrix = None

    for filepath in filepaths:
        binary_pixels = read_bw_png(filepath)
        if num_pixels is None:
            num_pixels = len(binary_pixels)
            weight_matrix = np.zeros((num_pixels, num_pixels))

        weight_matrix += np.outer(binary_pixels, binary_pixels)

    weight_matrix /= len(filepaths)
    np.fill_diagonal(weight_matrix, 0)

    return weight_matrix

def compute_threshold(weight_matrix):
    return np.sum(weight_matrix, axis=1) / 2

class HopfieldNetwork:
    def __init__(self, num_pixels):
        self.weight_matrix = None
        self.threshold = None
        self.state = None
        self.num_pixels = num_pixels
        self.iteration_energy = None
    
    def train(self, filepaths):
        self.weight_matrix = compute_weight_matrix(filepaths)
        self.threshold = compute_threshold(self.weight_matrix)

    def compute_energy(self):
        return -0.5 * np.dot(np.dot(self.state, self.weight_matrix), self.state) + np.dot(self.threshold, self.state)

    def update_neuron(self):
        global cancel_run, input_data
        value = np.dot(self.weight_matrix[self.i], self.state)
        energy = self.energy[-1]
        if value > self.threshold[self.i]:
            self.energy.append(energy - (1 - self.state[self.i]) * (value - self.threshold[self.i]))
            self.state[self.i] = 1
            canvas_image.put("#ffffff", (self.i % 64, self.i // 64))
        elif value < self.threshold[self.i]:
            self.energy.append(energy - (-1 - self.state[self.i]) * (value - self.threshold[self.i]))
            self.state[self.i] = -1
            canvas_image.put("#000000", (self.i % 64, self.i // 64))
        else:
            self.energy.append(energy)
        self.i += 1
        if self.i % 25 == 0:
            plot.set_data(range(len(self.energy)), self.energy)
            ax.set_xlim(0, len(self.energy))
            ax.set_ylim(min(self.energy), max(self.energy))
            graph.draw()
        if self.i == self.num_pixels:
            self.i = 0
            self.iterations += 1
            if self.energy[-1] != self.iteration_energy:
                self.iteration_energy = self.energy[-1]
            else:
                messagebox.showinfo("Converged", f"Converged after {self.iterations} iterations")
                cancel_run = True

        if cancel_run:
            input_data = self.state.copy()
            set_status("ready")
            return

        canvas.after(1, self.update_neuron)

    def update_neuron_instant(self):
        global cancel_run, input_data
        while True:
            for i in range(self.num_pixels):
                value = np.dot(self.weight_matrix[i], self.state)
                energy = self.energy[-1]
                if value > self.threshold[i]:
                    self.energy.append(energy - (1 - self.state[i]) * (value - self.threshold[i]))
                    self.state[i] = 1
                elif value < self.threshold[i]:
                    self.energy.append(energy - (-1 - self.state[i]) * (value - self.threshold[i]))
                    self.state[i] = -1
                else:
                    self.energy.append(energy)

            self.iterations += 1
            if self.energy[-1] != self.iteration_energy:
                self.iteration_energy = self.energy[-1]
            else:
                messagebox.showinfo("Converged", f"Converged after {self.iterations} iterations")
                cancel_run = True

            if cancel_run:
                cancel_run = False
                input_data = self.state.copy()
                for i in range(len(input_data)):
                    if input_data[i] == 1:
                        canvas_image.put("#ffffff", (i % 64, i // 64))
                    else:
                        canvas_image.put("#000000", (i % 64, i // 64))
                plot.set_data(range(len(self.energy)), self.energy)
                ax.set_xlim(0, len(self.energy))
                ax.set_ylim(min(self.energy), max(self.energy))
                graph.draw()
                set_status("ready")
                return

    def run(self, x):
        global cancel_run
        cancel_run = False
        self.iterations = 0
        self.i = 0
        self.state = x.copy()
        self.iteration_energy = self.compute_energy()
        self.energy = [self.iteration_energy]
        set_status("running")
        root.update()
        if instant_mode.get():
            self.update_neuron_instant()
        else:
            self.update_neuron()

def open_input_file():
    global input_data
    filepath = input_file_entry.get()
    if filepath:
        input_data = read_bw_png(filepath)
        for i in range(len(input_data)):
            if input_data[i] == 1:
                canvas_image.put("#ffffff", (i % 64, i // 64))
            else:
                canvas_image.put("#000000", (i % 64, i // 64))

def set_text(entry, text):
    entry.delete(0, tk.END)
    entry.insert(0, text)

def canvas_draw_white(e):
    if e.x < 0 or e.x > 63 or e.y < 0 or e.y > 63:
        return
    canvas_image.put("#ffffff", (e.x, e.y))
    input_data[e.y * 64 + e.x] = 1

def canvas_draw_black(e):
    if e.x < 0 or e.x > 63 or e.y < 0 or e.y > 63:
        return
    canvas_image.put("#000000", (e.x, e.y))
    input_data[e.y * 64 + e.x] = -1

def set_status(status):
    if status == "ready":
        status_label["text"] = "Status: Ready"
        save_button["state"] = tk.NORMAL
        run_button["state"] = tk.NORMAL
        open_file_button["state"] = tk.NORMAL
    elif status == "running":
        status_label["text"] = "Status: Running"
        save_button["state"] = tk.DISABLED
        run_button["state"] = tk.DISABLED
        open_file_button["state"] = tk.DISABLED

def set_cancel():
    global cancel_run
    cancel_run = True

root = tk.Tk()
root.title("Hopfield Network Image Autoassociation")

img_frame = tk.Frame(root, width=200, height=400, bg='grey')
img_frame.grid(row=0, column=0, padx=10, pady=5)

ui_frame = tk.Frame(root, width=650, height=400, bg='grey')
ui_frame.grid(row=0, column=1, padx=10, pady=5)

tk.Label(ui_frame, text="Input file:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
input_file_entry = tk.Entry(ui_frame, width=26)
input_file_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W, columnspan=2)
input_file_button = tk.Button(ui_frame, text="Choose file")
input_file_button.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
input_file_button["command"] = lambda: set_text(input_file_entry, filedialog.askopenfilename())
open_file_button = tk.Button(ui_frame, text="Load file", command=open_input_file)
open_file_button.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)
run_button = tk.Button(ui_frame, text="Run")
run_button.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
stop_button = tk.Button(ui_frame, text="Stop", command=set_cancel)
stop_button.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
save_button = tk.Button(ui_frame, text="Save file")
save_button.grid(row=1, column=2, padx=10, pady=5, sticky=tk.W)
save_button["command"] = lambda: canvas_image.write(filedialog.asksaveasfilename(defaultextension=".png"))
status_label = tk.Label(ui_frame, text="Status: Ready")
status_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
tk.Label(ui_frame, text="Instant mode:").grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
instant_mode = tk.BooleanVar()
instant_mode_check = tk.Checkbutton(ui_frame, variable=instant_mode)
instant_mode_check.grid(row=3, column=2, padx=10, pady=5, sticky=tk.W)

canvas = tk.Canvas(img_frame, width=64, height=64)
canvas.pack(side=tk.TOP)
canvas_image = tk.PhotoImage(width=64, height=64)
canvas.create_image(0, 0, anchor=tk.NW, image=canvas_image)
for i in range(len(input_data)):
    canvas_image.put("#ffffff", (i % 64, i // 64))
canvas.bind('<B1-Motion>', canvas_draw_black)
canvas.bind('<B3-Motion>', canvas_draw_white)

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
graph = FigureCanvasTkAgg(fig, master=img_frame)
graph.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
plot = ax.plot([], [], color="blue")[0]
ax.set_xlabel("Iteration")
ax.set_ylabel("Energy")

folder_path = "data"
filepaths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]
network = HopfieldNetwork(64 * 64)
network.train(filepaths)
run_button["command"] = lambda: network.run(input_data)

root.mainloop()
