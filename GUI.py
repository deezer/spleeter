import tkinter as tk
from tkinter import ttk
import subprocess

# Command for spleeter separate
command = "spleeter separate"

# Create a Tkinter window
window = tk.Tk()
window.title("Spleeter GUI")

# Function to execute the command
def execute_command():
    # Get the values from the GUI elements
    input_files = entry_files.get().split()
    codec = combo_codec.get()
    mwf = var_mwf.get()
    duration = entry_duration.get()
    offset = entry_offset.get()
    output_path = entry_output_path.get()
    stft_backend = var_stft_backend.get()
    filename_format = entry_filename_format.get()
    params_filename = entry_params_filename.get()
    verbose = var_verbose.get()

    # Construct the command with the selected options
    full_command = f"{command} {' '.join(input_files)} --codec {codec}"
    if mwf:
        full_command += " --mwf"
    if duration:
        full_command += f" --duration {duration}"
    if offset:
        full_command += f" --offset {offset}"
    if output_path:
        full_command += f" --output_path {output_path}"
    if stft_backend:
        full_command += f" --stft-backend {stft_backend}"
    if filename_format:
        full_command += f" --filename_format {filename_format}"
    if params_filename:
        full_command += f" --params_filename {params_filename}"
    if verbose:
        full_command += " --verbose"

    # Execute the command using subprocess
    subprocess.run(full_command, shell=True)

# Input files
label_files = tk.Label(window, text="Input Files:")
label_files.pack()
entry_files = tk.Entry(window, width=50)
entry_files.pack()

# Codec selection
label_codec = tk.Label(window, text="Audio Codec:")
label_codec.pack()
var_codec = tk.StringVar()
combo_codec = ttk.Combobox(window, textvariable=var_codec)
combo_codec['values'] = ('wav', 'mp3', 'ogg', 'm4a', 'wma', 'flac')
combo_codec.pack()

# MWF checkbox
var_mwf = tk.BooleanVar()
check_mwf = tk.Checkbutton(window, text="Use Multichannel Wiener Filtering", variable=var_mwf, onvalue=True, offvalue=False)
check_mwf.pack()

# Duration entry
label_duration = tk.Label(window, text="Duration:")
label_duration.pack()
entry_duration = tk.Entry(window)
entry_duration.pack()

# Offset entry
label_offset = tk.Label(window, text="Offset:")
label_offset.pack()
entry_offset = tk.Entry(window)
entry_offset.pack()

# Output path entry
label_output_path = tk.Label(window, text="Output Path:")
label_output_path.pack()
entry_output_path = tk.Entry(window)
entry_output_path.pack()

# STFT backend selection
label_stft_backend = tk.Label(window, text="STFT Backend:")
label_stft_backend.pack()
var_stft_backend = tk.StringVar()
combo_stft_backend = ttk.Combobox(window, textvariable=var_stft_backend)
combo_stft_backend['values'] = ('auto', 'tensorflow', 'librosa')
combo_stft_backend.pack()

# Filename format entry
label_filename_format = tk.Label(window, text="Filename Format:")
label_filename_format.pack()
entry_filename_format = tk.Entry(window)
entry_filename_format.pack()

# Params filename entry
label_params_filename = tk.Label(window, text="Params Filename:")
label_params_filename.pack()
entry_params_filename = tk.Entry(window)
entry_params_filename.pack()

# Verbose checkbox
var_verbose = tk.BooleanVar()
check_verbose = tk.Checkbutton(window, text="Verbose", variable=var_verbose, onvalue=True, offvalue=False)
check_verbose.pack()

# Button to execute the command
button_execute = tk.Button(window, text="Execute", command=execute_command)
button_execute.pack()

# Start the Tkinter event loop
window.mainloop()
