import re
import tkinter as tk
from tkinter import filedialog
import sys
from datetime import datetime

from constants import DEBUG
from Interpreter import Interpreter


# Instantiate the interpreter
interpreter = Interpreter()

# -----------------------------
# GUI for File Selection
# -----------------------------
def main():
    time = ""
    if DEBUG:
        print("main: Starting PadScript Interpreter GUI.")
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select PadScript File",
        filetypes=[("PadScript Files", "*.ps"), ("All Files", "*.*")]
    )
    if filename:
        if DEBUG:
            print(f"main: Selected file '{filename}'")
        time = datetime.now()
        interpreter.run(filename)
        time = datetime.now() - time
    else:
        if DEBUG:
            print("main: No file selected.")
    return time

# Entry point
if __name__ == "__main__":
    time = main()
    print(f"\nExecution time: {time.total_seconds()} seconds")
    #press any key to exit
    input("Press Enter to exit...")
    sys.exit(0)
