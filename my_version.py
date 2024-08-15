import cv2
import torch as pyt
import torch.nn as nn
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLineEdit, \
    QSizePolicy, QMainWindow
from PyQt5.QtGui import QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from qdarktheme import setup_theme
import time
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

'''
***MAIN(RUN) FILE*** - Introduced Changes from original_colors.py
Goals:
    Removed Unused Imports: Cleaned up the import statements.
    Improved Function Docstrings and Comments: I like readability for future lol.
    Efficient Data Handling: Used list comprehensions and in-place operations where possible.
    Optimized Training Loop: Reduced redundancy and optimized loss calculation and backpropagation.
    Modularized Code: Split larger functions into smaller, manageable parts.
    Improved Random Choice Handling: Used replace=False to avoid unnecessary checks.
    Added some exception handling for buttons
    Changed to %

    Code all HTML, CSS, & JAVASCRIPT in VsCode then bring into project
    Add button to PyQt5 interface to open a window to display HTML
    HTML is there in order to show irl examples (walls, outfits, etc)
    ^ For this used color theory with weighted preference of users colors
'''

# Constants
MODEL_ARCH = [64, 64, 32, 32, 16, 16, 8, 4]
OPTIM = pyt.optim.Adam
LOSS_FN = nn.MSELoss()
EPOCHS = 5
MIN_DIST = 20
LR = 1e-3
N_SUBDIVISIONS = 1024


def plot_color_gradients(cmap_category, cmap_list):
    """Plot color gradients for visualization."""
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

    for ax in axs:
        ax.set_axis_off()

    fig.show()


def gen_color():
    """Generate a random color and its corresponding color spaces."""
    hsvcolor = (np.random.rand(3) * np.array([360, 1, 1])).astype(np.float32).reshape(1, 1, 3)
    rgbcolor = cv2.cvtColor(hsvcolor, cv2.COLOR_HSV2RGB)
    labcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2LAB)
    return (rgbcolor.squeeze() * np.array([255, 255, 255])).astype(np.int32).tolist(), labcolor.squeeze(), pyt.tensor(
        (hsvcolor.squeeze() / np.array([360, 1, 1])).tolist() + rgbcolor.squeeze().tolist() + (
                    labcolor.squeeze() / np.array([100, 256, 256]) + np.array([0, 0.5, 0.5])).tolist())


def get_color(r, g, b):
    """Convert RGB color to different color spaces and return as tensor."""
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


class ColorGuesserManager:
    def __init__(self):
        self.reset_colors()
        self.data = []
        self.model = ColorGuesser(MODEL_ARCH)
        self.optim = OPTIM(self.model.parameters(), lr=LR)

    def reset_colors(self):
        self.color1 = gen_color()
        self.color2 = gen_color()
        while np.linalg.norm(self.color1[1] - self.color2[1]) < MIN_DIST:
            self.color2 = gen_color()

    def run_iter(self, choice):
        print(f"Adding data point with choice: {choice}")
        self.data.append((pyt.cat((self.color1[2], self.color2[2])), choice))
        self.reset_colors()
        print(f"Data length: {len(self.data)}")
        if len(self.data) % 10 == 0:
            if len(self.data) > 20:
                training_data = self.data[-10:]
                indices = np.random.choice(len(self.data) - 10, 10, replace=False)
                training_data.extend(self.data[i] for i in indices)
            else:
                training_data = self.data
            print(f"Training data length: {len(training_data)}")
            return self.train_loop(training_data)
        return None

    def train_loop(self, training_data):
        if not training_data:
            return "No training data available."

        start_time = time.time()
        total_loss = 0

        for epoch in range(EPOCHS):
            for input, target in training_data:
                self.optim.zero_grad()
                output = self.model(input)
                loss = LOSS_FN(output, target)
                total_loss += loss.item()
                loss.backward()
                self.optim.step()

        elapsed_time = time.time() - start_time
        correct = 0

        with pyt.no_grad():
            eval_data = self.data if len(self.data) <= 100 else [self.data[i] for i in np.random.choice(len(self.data), 100, replace=False)]
            for input, target in eval_data:
                output = self.model(input)
                correct += pyt.argmax(output) == pyt.argmax(target)

        avg_loss = total_loss / (len(training_data) * EPOCHS)
        accuracy = correct / len(eval_data)
        print(f"Training completed. Avg Loss: {avg_loss}, Accuracy: {accuracy}")
        return f"Accuracy: {accuracy:.2%}\nAverage Loss: {avg_loss}\nTraining Time: {elapsed_time * 1000:.2f} ms"

    def eval_model(self, c1, c2):
        if pyt.equal(c1, c2):
            return False
        return pyt.argmax(self.model(pyt.cat((c1, c2)))) == 1

    def csort(self, arr, fst, lst):
        if fst >= lst:
            return
        pivot = arr[np.random.randint(fst, lst)]
        i, j = fst, lst
        while i <= j:
            while self.eval_model(pivot[1], arr[i][1]):
                i += 1
            while self.eval_model(arr[j][1], pivot[1]):
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        self.csort(arr, fst, j)
        self.csort(arr, i, lst)

    def sort_colors(self):
        start_time = time.time()
        colors = [([r * 15, g * 15, b * 15], get_color(r * 15, g * 15, b * 15)) for r in range(18) for g in range(18) for b in range(18)]
        self.csort(colors, 0, len(colors) - 1)
        elapsed_time = time.time() - start_time
        print(f"Sorted list of colors in {elapsed_time:.2f} seconds!")

        # Unpack colors for visualization
        unpacked_colors = [(*rgb, 255) for rgb, _ in colors]
        rlist = [next(c for c in unpacked_colors if c[0] == r * 15) for r in range(18)]
        glist = [next(c for c in unpacked_colors if c[1] == g * 15) for g in range(18)]
        blist = [next(c for c in unpacked_colors if c[2] == b * 15) for b in range(18)]

        return np.array(unpacked_colors) / 255, np.array(rlist) / 255, np.array(glist) / 255, np.array(blist) / 255


class ColorChoice(QWidget):
    def __init__(self):
        super().__init__()
        self.Guesser = ColorGuesserManager()
        self.initUI()
        self.path = os.path.join(os.path.dirname(__file__), "models")

    def initUI(self):
        self.setWindowTitle("Color Choice")
        self.colors = [QColor(*self.Guesser.color1[0]), QColor(*self.Guesser.color2[0])]
        self.selected_color_index = -1

        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.buttons = [QPushButton(self) for _ in range(2)]
        for i, button in enumerate(self.buttons):
            button.setStyleSheet(f"background-color: {self.colors[i].name()};")
            button.clicked.connect(lambda _, index=i: self.color_clicked(index))
            button.setMinimumSize(200, 100)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)

        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumSize(400, 100)
        self.stats_box.setText("Accuracy after 0 iterations: 0.00%")
        layout.addWidget(self.stats_box)

        eval_retrain_layout = QHBoxLayout()
        self.eval_button = QPushButton("Eval")
        self.eval_button.clicked.connect(self.eval_clicked)
        eval_retrain_layout.addWidget(self.eval_button)

        self.open_window_button = QPushButton("Open Visualizer")
        self.open_window_button.clicked.connect(self.open_html_window)
        eval_retrain_layout.addWidget(self.open_window_button)

        self.retrain_button = QPushButton("Retrain")
        self.retrain_button.clicked.connect(self.retrain_clicked)
        eval_retrain_layout.addWidget(self.retrain_button)
        layout.addLayout(eval_retrain_layout)

        save_load_layout = QHBoxLayout()
        self.name_text = QLineEdit()
        self.name_text.setPlaceholderText("Model Name")
        save_load_layout.addWidget(self.name_text)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_clicked)
        save_load_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_clicked)
        save_load_layout.addWidget(self.load_button)

        layout.addLayout(save_load_layout)
        self.setLayout(layout)
        self.setMinimumSize(800, 250)
        self.show()

    def color_clicked(self, index):
        choice = [0, 0]
        choice[index] = 1
        result = self.Guesser.run_iter(pyt.tensor(choice, dtype=pyt.float32))
        self.colors = [QColor(*self.Guesser.color1[0]), QColor(*self.Guesser.color2[0])]
        for i, button in enumerate(self.buttons):
            button.setStyleSheet(f"background-color: {self.colors[i].name()};")
        if result:
            self.stats_box.setText(result)

    def save_clicked(self):
        path = os.path.join(self.path, self.name_text.text())
        pyt.save(self.Guesser.model.state_dict(), path + "model.pt")
        pyt.save(self.Guesser.optim.state_dict(), path + "optim.pt")
        pyt.save(self.Guesser.data, path + "data.pt")

    def load_clicked(self):
        try:
            path = os.path.join(self.path, self.name_text.text())
            self.Guesser.model.load_state_dict(pyt.load(path + "model.pt"))
            self.Guesser.optim.load_state_dict(pyt.load(path + "optim.pt"))
            self.Guesser.data = pyt.load(path + "data.pt")
            self.stats_box.setText("Model and data loaded successfully.")
        except Exception as e:
            self.stats_box.setText(f"Error: {str(e)}")

    def eval_clicked(self):
        try:
            c, r, g, b = self.Guesser.sort_colors()
            plot_color_gradients("Color Preferences:", [["Overall", c], ["Red", r], ["Green", g], ["Blue", b]])
        except Exception as e:
            self.stats_box.setText(f"Error: {str(e)}")

    def open_html_window(self):
        # SET COLOR HERE (I think?)
        self.html_window = HtmlWindow()
        self.html_window.show()

    def retrain_clicked(self):
        result = self.Guesser.train_loop(self.Guesser.data)
        self.stats_box.setText(result)


class HtmlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizer")

        self.web_view = QWebEngineView()

        html_file = os.path.join(os.path.dirname(__file__), 'HTML/index.html')
        self.web_view.setUrl(QUrl.fromLocalFile(html_file))

        layout = QVBoxLayout()
        layout.addWidget(self.web_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


if __name__ == '__main__':
    app = QApplication([])
    setup_theme("auto")
    window = ColorChoice()
    app.exec_()
