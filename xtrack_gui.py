import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from xtrack import *
from utils import *

# Function to process files in the input folder and put results in the output folder
def process_files(input_folder, output_folder, variable_values):
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):  # Change the file extension according to your needs
            with open(os.path.join(input_folder, filename), 'r') as file:
                # Example processing: Changing variable values in each file
                filename, waveform, spectrogram_np, probs = processAudioFile(input_folder+'/'+filename)
                predicted_onsets, predicted_labels = Xtrack(probs, MUSIC_START_OFFSET=variable_values[0], MUSIC_STOP_OFFSET=variable_values[1])
        
                output_path = output_folder+'/'+filename+'/'
                create_directory(output_path)

                writeIndividualTracks(waveform, predicted_onsets, predicted_labels, output_path=output_path)
                writeIndexesCSV(predicted_onsets, predicted_labels, output_path=output_path)

# Function to browse and select folders
def browse_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, folder_path)

# Function to execute the code when the button is clicked
def run_code():
    input_folder = input_entry.get()
    output_folder = output_entry.get()
    variable_values = [var1_slider.get(), var2_slider.get()]#, var3_slider.get(), var4_slider.get()]
    process_files(input_folder, output_folder, variable_values)

# Function to update the label text with the current slider value
def update_label(slider, label):
    label.config(text=f'{slider.get():.2f}')

# Create the main window
root = tk.Tk()
root.title("XTrack: audio segmentation")

# Input folder selection
tk.Label(root, text="Data Folder:").grid(row=0, column=0)
input_entry = tk.Entry(root, width=50)
input_entry.grid(row=0, column=1)
input_button = tk.Button(root, text="Browse", command=lambda: browse_folder(input_entry))
input_button.grid(row=0, column=2)

# Output folder selection
tk.Label(root, text="Output Folder:").grid(row=1, column=0)
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=1, column=1)
output_button = tk.Button(root, text="Browse", command=lambda: browse_folder(output_entry))
output_button.grid(row=1, column=2)

# Default values for the variables
default_values = [-0.1, 4, 0.5, 0.5]

# Variable 1 slider
tk.Label(root, text="Before start offset:").grid(row=2, column=0)
var1_slider = ttk.Scale(root, from_=-5, to=0, length=200, value=default_values[0], orient='horizontal', command=lambda value: update_label(var1_slider, var1_label))
var1_slider.grid(row=2, column=1)
var1_label = tk.Label(root, text=f'{default_values[0]:.2f}')
var1_label.grid(row=2, column=2)

# Variable 2 slider
tk.Label(root, text="Ater stop offset:").grid(row=3, column=0)
var2_slider = ttk.Scale(root, from_=0, to=10, length=200, value=default_values[1], orient='horizontal', command=lambda value: update_label(var2_slider, var2_label))
var2_slider.grid(row=3, column=1)
var2_label = tk.Label(root, text=f'{default_values[1]:.2f}')
var2_label.grid(row=3, column=2)

# # Variable 3 slider
# tk.Label(root, text="Variable 3:").grid(row=4, column=0)
# var3_slider = ttk.Scale(root, from_=0, to=1, length=200, value=default_values[2], orient='horizontal', command=lambda value: update_label(var3_slider, var3_label))
# var3_slider.grid(row=4, column=1)
# var3_label = tk.Label(root, text=f'{default_values[2]:.2f}')
# var3_label.grid(row=4, column=2)

# # Variable 4 slider
# tk.Label(root, text="Variable 4:").grid(row=5, column=0)
# var4_slider = ttk.Scale(root, from_=0, to=1, length=200, value=default_values[3], orient='horizontal', command=lambda value: update_label(var4_slider, var4_label))
# var4_slider.grid(row=5, column=1)
# var4_label = tk.Label(root, text=f'{default_values[3]:.2f}')
# var4_label.grid(row=5, column=2)

# Run button
run_button = tk.Button(root, text="Run Code", command=run_code)
run_button.grid(row=6, column=1)

# Run the main event loop
root.mainloop()