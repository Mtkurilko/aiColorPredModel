import streamlit as st
import torch as pyt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import cv2
import time

def plot_color_gradients(cmap_category, cmap_list):
    """Plot color gradients for visualization."""
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=0.978, bottom=0.000, left=0.030, right=1.000)

    axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

    for ax, cmap in zip(axs, cmap_list):
        gradient = np.linspace(0, 1, 1024)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto',
                  cmap=LinearSegmentedColormap.from_list(cmap[0], colors=cmap[1], N=1024))
        ax.text(-.01, .5, cmap[0], va='center', ha='right', fontsize=10, transform=ax.transAxes)

    for ax in axs:
        ax.set_axis_off()

    st.pyplot(fig)

def gen_color():
    """Generate a random color and its corresponding color spaces."""
    hsvcolor = (np.random.rand(3) * np.array([360, 1, 1])).astype(np.float32).reshape(1, 1, 3)
    rgbcolor = cv2.cvtColor(hsvcolor, cv2.COLOR_HSV2RGB)
    labcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2LAB)
    rgbcolor = (rgbcolor.squeeze() * np.array([255, 255, 255])).astype(np.int32)
    return rgbcolor.tolist(), labcolor.squeeze(), pyt.tensor(
        (hsvcolor.squeeze() / np.array([360, 1, 1])).tolist() + rgbcolor.tolist() + (
                    labcolor.squeeze() / np.array([100, 256, 256]) + np.array([0, 0.5, 0.5])).tolist())

def get_color(r, g, b):
    """Convert RGB color to different color spaces and return as tensor."""
    rgbcolor = np.array([r, g, b]).astype(np.float32).reshape(1, 1, 3)
    hsvcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2HSV)
    labcolor = cv2.cvtColor(rgbcolor, cv2.COLOR_RGB2LAB)
    return pyt.tensor((hsvcolor.squeeze() / np.array([360, 1, 1])).tolist() + rgbcolor.squeeze().tolist() + (
                labcolor.squeeze() / np.array([100, 256, 256]) + np.array([0, 0.5, 0.5])).tolist(), dtype=pyt.float32)

class ColorGuesser(pyt.nn.Module):
    def __init__(self, mid_layers):
        super().__init__()
        layers = [pyt.nn.Linear(18, mid_layers[0]), pyt.nn.LeakyReLU()]
        for i in range(len(mid_layers) - 1):
            layers.append(pyt.nn.Linear(mid_layers[i], mid_layers[i + 1]))
            layers.append(pyt.nn.LeakyReLU())
        layers.append(pyt.nn.Linear(mid_layers[-1], 2))
        layers.append(pyt.nn.Softmax(dim=-1))
        self.model = pyt.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ColorGuesserManager:
    def __init__(self):
        self.reset_colors()
        self.data = []
        self.model = ColorGuesser([64, 64, 32, 32, 16, 16, 8, 4])
        self.optim = pyt.optim.Adam(self.model.parameters(), lr=1e-3)
        self.last_accuracy = None
        self.last_loss = None

    def reset_colors(self):
        self.color1 = gen_color()
        self.color2 = gen_color()
        while np.linalg.norm(self.color1[1] - self.color2[1]) < 20:
            self.color2 = gen_color()

    def run_iter(self, choice):
        if choice is None:
            return "No choice selected."

        # Convert choice tensor to Python list and pick the first element
        choice = choice.tolist()
        if len(choice) != 2 or not (0 <= choice[0] <= 1 and 0 <= choice[1] <= 1):
            return "Invalid choice."

        self.data.append((pyt.cat((self.color1[2], self.color2[2])), pyt.tensor(choice)))
        self.reset_colors()

        if len(self.data) % 10 == 0:
            if len(self.data) > 20:
                training_data = self.data[-10:]
                indices = np.random.choice(len(self.data) - 10, 10, replace=False)
                training_data.extend(self.data[i] for i in indices)
            else:
                training_data = self.data
            result = self.train_loop(training_data)
            return result
        return None

    def train_loop(self, training_data):
        if not training_data:
            return "No training data available."

        start_time = time.time()
        total_loss = 0

        for epoch in range(5):
            for input, target in training_data:
                self.optim.zero_grad()
                output = self.model(input)
                loss = pyt.nn.MSELoss()(output, target)
                total_loss += loss.item()
                loss.backward()
                self.optim.step()

        elapsed_time = time.time() - start_time
        correct = 0

        with pyt.no_grad():
            eval_data = self.data if len(self.data) <= 100 else [self.data[i] for i in
                                                                 np.random.choice(len(self.data), 100, replace=False)]
            for input, target in eval_data:
                output = self.model(input)
                correct += pyt.argmax(output) == pyt.argmax(target)

        avg_loss = total_loss / (len(training_data) * 5)
        accuracy = correct / len(eval_data)
        self.last_accuracy = accuracy
        self.last_loss = avg_loss
        return f"Accuracy: {accuracy:.2%}\nAverage Loss: {avg_loss}\nTraining Time: {elapsed_time * 1000:.2f} ms"

    def eval_model(self, c1, c2):
        if pyt.equal(c1, c2).all().item():  # Convert tensor comparison to boolean
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
        colors = [([r * 15, g * 15, b * 15], get_color(r * 15, g * 15, b * 15)) for r in range(18) for g in range(18)
                  for b in range(18)]
        self.csort(colors, 0, len(colors) - 1)
        elapsed_time = time.time() - start_time

        # Unpack colors for visualization
        unpacked_colors = [(*rgb, 255) for rgb, _ in colors]
        rlist = [next(c for c in unpacked_colors if c[0] == r * 15) for r in range(18)]
        glist = [next(c for c in unpacked_colors if c[1] == g * 15) for g in range(18)]
        blist = [next(c for c in unpacked_colors if c[2] == b * 15) for b in range(18)]

        return np.array(unpacked_colors) / 255, np.array(rlist) / 255, np.array(glist) / 255, np.array(blist) / 255

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def main():
    st.set_page_config(page_title="🎨 Color Choice", layout="wide")
    st.title("🎨 Color Choice")

    guesser = ColorGuesserManager()

    # Model Save/Load
    st.sidebar.header("Model Save/Load")
    model_name = st.sidebar.text_input("Model Name", "")

    save_load_cols = st.sidebar.columns(2)
    with save_load_cols[0]:
        if st.button("Save"):
            try:
                path = os.path.join(os.path.dirname(__file__), "models", model_name)
                os.makedirs(path, exist_ok=True)
                pyt.save(guesser.model.state_dict(), os.path.join(path, "model.pt"))
                pyt.save(guesser.optim.state_dict(), os.path.join(path, "optim.pt"))
                # Convert data to list format before saving
                pyt.save([([item[0].tolist(), item[1].tolist()]) for item in guesser.data], os.path.join(path, "data.pt"))
                st.success("Model and data saved successfully.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with save_load_cols[1]:
        if st.button("Load"):
            try:
                path = os.path.join(os.path.dirname(__file__), "models", model_name)
                guesser.model.load_state_dict(pyt.load(os.path.join(path, "model.pt")))
                guesser.optim.load_state_dict(pyt.load(os.path.join(path, "optim.pt")))
                # Load data from file and convert back to the original format
                loaded_data = pyt.load(os.path.join(path, "data.pt"))
                guesser.data = [(pyt.tensor(item[0]), pyt.tensor(item[1])) for item in loaded_data]
                st.success("Model and data loaded successfully.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display color choices
    col1, col2 = st.columns(2)
    if 'color1_selected' not in st.session_state:
        st.session_state.color1_selected = False
    if 'color2_selected' not in st.session_state:
        st.session_state.color2_selected = False
    if 'result' not in st.session_state:
        st.session_state.result = None

    with col1:
        st.markdown(
            f'<div style="background-color: {rgb_to_hex(guesser.color1[0])}; width: 200px; height: 200px; cursor: pointer;" '
            f'onclick="document.getElementById(\'color1_button\').click();"></div>', unsafe_allow_html=True)
        st.button("Select Color 1", key="color1_button", on_click=lambda: st.session_state.update({
            'result': guesser.run_iter(pyt.tensor([1, 0], dtype=pyt.float32)),
            'color1_selected': True,
            'color2_selected': False
        }), use_container_width=True)

    with col2:
        st.markdown(
            f'<div style="background-color: {rgb_to_hex(guesser.color2[0])}; width: 200px; height: 200px; cursor: pointer;" '
            f'onclick="document.getElementById(\'color2_button\').click();"></div>', unsafe_allow_html=True)
        st.button("Select Color 2", key="color2_button", on_click=lambda: st.session_state.update({
            'result': guesser.run_iter(pyt.tensor([0, 1], dtype=pyt.float32)),
            'color1_selected': False,
            'color2_selected': True
        }), use_container_width=True)

    # Display current colors and results
    if st.session_state.result:
        st.write(st.session_state.result)

    # Accuracy and loss display
    if guesser.last_accuracy is not None and guesser.last_loss is not None:
        st.session_state.accuracy_display = f"Accuracy: {guesser.last_accuracy:.2%}\nAverage Loss: {guesser.last_loss}"
    else:
        st.session_state.accuracy_display = "No data available."

    st.sidebar.text_area("Model Performance", st.session_state.accuracy_display, height=100)

    # Evaluation and retrain buttons
    if st.session_state.color1_selected or st.session_state.color2_selected:
        if st.button("Eval"):
            st.session_state.result = guesser.train_loop(guesser.data)
        if st.button("Retrain"):
            st.session_state.result = guesser.train_loop(guesser.data)

    # Show color gradients
    try:
        c, r, g, b = guesser.sort_colors()
        plot_color_gradients("Color Preferences:", [["Overall", c], ["Red", r], ["Green", g], ["Blue", b]])
    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()