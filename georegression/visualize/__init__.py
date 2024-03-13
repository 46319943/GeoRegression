from pathlib import Path
import matplotlib.pyplot as plt

# Create folder for saving plot
default_folder = Path('Plot')
default_folder.mkdir(exist_ok=True)

# Set default style for matplotlib

# Control style to make it consistent with the Plotly
# plt.style.use('seaborn')
plt.style.use('seaborn-v0_8')
# Font family
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
