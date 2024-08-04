import cv2
import torch as pyt
import torch.nn as nn
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QSizePolicy, \
    QLineEdit
from PyQt5.QtGui import QColor
from qdarktheme import setup_theme
import time
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

MODEL_ARCH = [64, 64, 32, 32, 16, 16, 8, 4]
OPTIM = pyt.optim.Adam
LOSS_FN = nn.MSELoss()
EPOCHS = 5
MIN_DIST = 20
LR = 1e-3
N_SUBDIVISIONS = 1024


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=0.978, bottom=0.000, left=0.030, right=1.000)

    axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

    for ax, cmap in zip(axs, cmap_list):
        gradient = np.linspace(0, 1, N_SUBDIVISIONS)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto',
                  cmap=LinearSegmentedColormap.from_list(cmap[0], colors=cmap[1], N=N_SUBDIVISIONS))
        ax.text(-.01, .5, cmap[0], va='center', ha='right', fontsize=10, transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    fig.show()


def gen_color():
    hsvcolor = (np.random.rand(3) * np.array([360, 1, 1])).astype(np.float32).reshape(1, 1, 3)
    rgbcolor = cv2.cvtColor(hsvcolor, cv2.COLOR_HSV2RGB)
    labcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2LAB)
    return (rgbcolor.squeeze() * np.array([255, 255, 255])).astype(np.int32).tolist(), labcolor.squeeze(), pyt.tensor(
        (hsvcolor.squeeze() / np.array([360, 1, 1])).tolist() + rgbcolor.squeeze().tolist() + (
                    labcolor.squeeze() / np.array([100, 256, 256]) + np.array([0, 0.5, 0.5])).tolist())


def get_color(r, g, b):
    rgbcolor = np.array([r, g, b]).astype(np.float32).reshape(1, 1, 3)
    hsvcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2HSV)
    labcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2LAB)
    return pyt.tensor((hsvcolor.squeeze() / np.array([360, 1, 1])).tolist() + rgbcolor.squeeze().tolist() + (
                labcolor.squeeze() / np.array([100, 256, 256]) + np.array([0, 0.5, 0.5])).tolist(), dtype=pyt.float32)


class ColorGuesser(nn.Module):
    def __init__(self, mid_layers):
        super().__init__()
        layers = [nn.Linear(18, mid_layers[0]), nn.LeakyReLU()]
        for i in range(len(mid_layers) - 1):
            layers.append(nn.Linear(mid_layers[i], mid_layers[i + 1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(mid_layers[-1], 2))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ColorGuesserManager():
    def __init__(self):
        self.color1 = None
        self.color2 = None
        self.reset_colors()
        self.data = []
        self.model = ColorGuesser(MODEL_ARCH)
        self.optim = OPTIM(self.model.model.parameters(), lr=LR)

    def reset_colors(self):
        self.color1 = gen_color()
        self.color2 = gen_color()
        while np.linalg.norm(self.color1[1] - self.color2[1]) < MIN_DIST:
            self.color2 = gen_color()

    def run_iter(self, choice):
        self.data.append((pyt.cat((self.color1[2], self.color2[2])), choice))
        self.reset_colors()
        if len(self.data) % 10 == 0:
            if len(self.data) > 20:
                training_data = self.data[-10:]
                indicies = np.random.choice(len(self.data) - 10, 10)
                for i in indicies:
                    training_data.append(self.data[i])
            else:
                training_data = self.data
            return self.train_loop(training_data)
        return None

    def train_loop(self, training_data):
        t1 = time.time()
        total_loss = 0
        for i in range(EPOCHS):
            for input, target in training_data:
                self.optim.zero_grad()
                output = self.model(input)
                loss = LOSS_FN(output, target)
                total_loss += loss.item()
                loss.backward()
                self.optim.step()
        t2 = time.time()
        correct = 0
        with pyt.no_grad():
            eval_data = []
            if len(self.data) > 100:
                indicies = np.random.choice(len(self.data), 100)
                for i in indicies:
                    eval_data.append(self.data[i])
            else:
                eval_data = self.data
            for input, target in eval_data:
                output = self.model(input)
                correct += 1 if pyt.argmax(output, dim=-1).item() == pyt.argmax(target, dim=-1).item() else 0
            avg_loss = total_loss / (len(training_data) * EPOCHS)
            return f"Accuracy after {len(self.data)} iterations: {correct}/{len(eval_data)}\nAverage Loss: {avg_loss}\nTraining time: {(t2 - t1) * 1000:.2f}ms\nEval Time: {(time.time() - t2) * 1000:.2f}ms"

    def eval_model(self, c1, c2):
        if pyt.equal(c1, c2):
            return False
        return pyt.argmax(self.model(pyt.concat((c1, c2))), dim=-1).item() == 1

    def csort(self, arr, fst, lst):
        print(f"Sorting! First: {fst}, Last: {lst}")
        if fst >= lst:
            return arr
        i, j = fst, lst
        p = arr[np.random.randint(fst, lst)]
        while i <= j:
            while self.eval_model(p[1], arr[i][1]):
                i += 1
            while self.eval_model(arr[j][1], p[1]):
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i, j = i + 1, j - 1
        self.csort(arr, fst, j)
        self.csort(arr, i, lst)

    def sort_colors(self):
        t1 = time.time()
        colors = []
        for r in range(18):
            for g in range(18):
                for b in range(18):
                    colors.append(([r * 15, g * 15, b * 15], get_color(r * 15, g * 15, b * 15)))
        self.csort(colors, 0, len(colors) - 1)
        colors = [(i[0][0], i[0][1], i[0][2], 255) for i in colors]
        rlist, glist, blist = [], [], []
        for r in range(18):
            rlist.append(next(c for c in colors if c[0] == r * 15))
        for g in range(18):
            glist.append(next(c for c in colors if c[1] == g * 15))
        for b in range(18):
            blist.append(next(c for c in colors if c[2] == b * 15))
        print(f"\nSorted list of colors in {time.time() - t1:.2f} seconds!")
        colors, rlist, glist, blist = np.array(colors, dtype=np.float32) / 255, np.array(rlist,
                                                                                         dtype=np.float32) / 255, np.array(
            glist, dtype=np.float32) / 255, np.array(blist, dtype=np.float32) / 255
        return colors, rlist, glist, blist


class ColorChoice(QWidget):
    def __init__(self):
        super().__init__()
        self.Guesser = ColorGuesserManager()
        self.initUI()
        self.path = os.path.dirname(__file__) + "//models//"

    def initUI(self):
        self.setWindowTitle("Color Choice")

        # Define initial colors
        self.colors = [QColor(*self.Guesser.color1[0]), QColor(*self.Guesser.color2[0])]
        self.selected_color_index = -1

        # Layout
        layout = QVBoxLayout()

        # Buttons
        button_layout = QHBoxLayout()
        self.buttons = [QPushButton(self) for _ in range(2)]
        for i, button in enumerate(self.buttons):
            button.setStyleSheet(f"background-color: {self.colors[i].name()};")
            button.clicked.connect(lambda checked, index=i: self.color_clicked(index))
            button.setMinimumSize(200, 100)
            size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button.setSizePolicy(size_policy)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)

        # Stats textbox
        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumSize(400, 100)
        self.stats_box.setText("Accuracy after 0 iterations: 0.00%")
        layout.addWidget(self.stats_box)

        # Eval and Retrain buttons
        evail_retrain_layout = QHBoxLayout()
        self.eval_button = QPushButton("Eval")
        self.eval_button.clicked.connect(self.eval_clicked)  # Connect to eval function
        evail_retrain_layout.addWidget(self.eval_button)

        self.retrain_button = QPushButton("Retrain")
        self.retrain_button.clicked.connect(self.retrain_clicked)  # Connect to eval function
        evail_retrain_layout.addWidget(self.retrain_button)

        layout.addLayout(evail_retrain_layout)

        # Save and load buttons
        save_load_layout = QHBoxLayout()

        self.name_text = QLineEdit()
        self.name_text.setPlaceholderText("Model Name")
        save_load_layout.addWidget(self.name_text)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_clicked)  # Connect to save function
        save_load_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_clicked)  # Connect to load function
        save_load_layout.addWidget(self.load_button)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data_clicked)  # Connect to load function
        save_load_layout.addWidget(self.load_button)

        layout.addLayout(save_load_layout)

        self.setMinimumSize(800, 250)
        self.setLayout(layout)
        self.show()

    def color_clicked(self, index):
        l = [0, 0]
        l[index] = 1
        text = self.Guesser.run_iter(pyt.tensor(l).to(pyt.float32))
        self.colors = [QColor(*self.Guesser.color1[0]), QColor(*self.Guesser.color2[0])]
        for i, button in enumerate(self.buttons):
            button.setStyleSheet(f"background-color: {self.colors[i].name()};")
        if type(text) == str:
            self.stats_box.setText(text)

    def save_clicked(self):
        path = self.path + self.name_text.text()
        pyt.save(self.Guesser.model.state_dict(), path + "model.pt")
        pyt.save(self.Guesser.optim.state_dict(), path + "optim.pt")
        pyt.save(self.Guesser.data, path + "data.pt")

    def load_clicked(self):
        path = self.path + self.name_text.text()
        self.Guesser.model.load_state_dict(pyt.load(path + "model.pt"))
        self.Guesser.optim.load_state_dict(pyt.load(path + "optim.pt"))
        self.Guesser.data = pyt.load(path + "data.pt")

    def load_data_clicked(self):
        path = self.path + self.name_text.text()
        self.Guesser.model = ColorGuesser(MODEL_ARCH)
        self.Guesser.optim = OPTIM(self.Guesser.model.model.parameters(), lr=LR)
        self.Guesser.data = pyt.load(path + "data.pt")
        self.stats_box.setText(self.Guesser.train_loop(self.Guesser.data))

    def eval_clicked(self):
        c, r, g, b = self.Guesser.sort_colors()
        plot_color_gradients("Color Preferences:", [["Overall", c], ["Red", r], ["Green", g], ["Blue", b]])

    def retrain_clicked(self):
        self.Guesser.model = ColorGuesser(MODEL_ARCH)
        self.Guesser.optim = OPTIM(self.Guesser.model.model.parameters(), lr=LR)
        self.stats_box.setText(self.Guesser.train_loop(self.Guesser.data))


if __name__ == '__main__':
    app = QApplication([])
    setup_theme("auto")
    window = ColorChoice()
    app.exec_()