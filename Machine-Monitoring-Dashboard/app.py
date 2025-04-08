import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import csv
import time
import threading
import pandas as pd
import os
import winsound
import openai
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------- #
# CONFIGURATIONS
SERIAL_PORT = "COM3"  
BAUD_RATE = 9600
SIMULATE = True  
CSV_FILE = "machine_status.csv"
EXCEL_FILE = "machine_status.xlsx"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ------------------------------- #

# Try to connect to Serial Port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    SIMULATE = False
except:
    print("No device found, using manual input mode.")

# Ensure CSV file has headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Binary Input", "Status", "AI Suggestion"])

# Function to generate AI-based repair suggestions
def generate_fault_solution(fault_type):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a machine repair expert. Provide a short fix for detected faults."},
                {"role": "user", "content": f"What is the best way to fix {fault_type}?"}
            ],
            max_tokens=100
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "AI error: Check API key & internet."

# Function to update machine status
def update_status(binary_string):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "Normal"
    fault_solution = ""

    if "1" in binary_string:
        status = "Faulty"
        status_label.config(text="⚠ Fault Detected!", fg="red", font=("Arial", 16, "bold"))
        winsound.Beep(1000, 500)  

        fault_solution = generate_fault_solution("Machine Fault Detected")
        messagebox.showwarning("⚠ Machine Fault", f"Fault detected!\nSuggested Fix:\n{fault_solution}")
    else:
        status_label.config(text="✅ Machine Running Smoothly", fg="green", font=("Arial", 16, "bold"))

    # Save to CSV file
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, binary_string, status, fault_solution])

    # Save to Excel
    save_to_excel(timestamp, binary_string, status, fault_solution)

    # Update table
    log_table.insert("", 0, values=(timestamp, binary_string, status, fault_solution))

    # Update graph
    update_graph()

# Function to save data in Excel
def save_to_excel(timestamp, binary_string, status, fault_solution):
    data = {"Timestamp": [timestamp], "Binary Input": [binary_string], "Status": [status], "AI Suggestion": [fault_solution]}
    df = pd.DataFrame(data)

    if os.path.exists(EXCEL_FILE):
        existing_data = pd.read_excel(EXCEL_FILE)
        df = pd.concat([existing_data, df], ignore_index=True)

    df.to_excel(EXCEL_FILE, index=False)

# Function to process user input
def process_input():
    binary_string = user_input.get().strip()
    if binary_string and all(c in "01" for c in binary_string):
        update_status(binary_string)
        user_input.delete(0, tk.END)
    else:
        status_label.config(text="❌ Invalid Input! Enter binary (e.g., 0001)", fg="orange")

# Function to read real-time data
def read_data():
    while True:
        if not SIMULATE:
            binary_string = ser.readline().decode().strip()
            if binary_string:
                update_status(binary_string)
        time.sleep(1)

# Function to export logs
def export_logs():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if file_path:
        if file_path.endswith(".csv"):
            with open(CSV_FILE, "r") as src, open(file_path, "w", newline="") as dst:
                dst.write(src.read())
            messagebox.showinfo("Export Successful", "Logs saved as CSV!")
        elif file_path.endswith(".xlsx"):
            os.rename(EXCEL_FILE, file_path)
            messagebox.showinfo("Export Successful", "Logs saved as Excel!")

# Function to update real-time graph
def update_graph():
    df = pd.read_csv(CSV_FILE)
    timestamps = df["Timestamp"][-20:]
    statuses = df["Status"].apply(lambda x: 1 if x == "Faulty" else 0)[-20:]

    ax.clear()
    ax.plot(timestamps, statuses, marker="o", linestyle="-", color="red", label="Faults Over Time")
    ax.set_xticklabels(timestamps, rotation=45, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Faulty"])
    ax.legend()
    canvas.draw()

# ------------------------------- #
# UI Setup
root = tk.Tk()
root.title("Machine Monitoring System")
root.geometry("900x600")

# Styling
style = ttk.Style()
style.configure("TButton", font=("Arial", 12, "bold"), padding=10, relief="flat")
style.configure("TLabel", font=("Arial", 14, "bold"))
style.configure("Treeview.Heading", font=("Arial", 12, "bold"))
style.configure("Treeview", font=("Arial", 12), rowheight=25)

# Main Frame
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill="both", expand=True)

# Status Label
status_label = tk.Label(main_frame, text="✅ Machine Running Smoothly", font=("Arial", 16, "bold"), fg="green")
status_label.pack(pady=10)

# Input Section
input_frame = tk.Frame(main_frame)
input_frame.pack()

user_input = tk.Entry(input_frame, font=("Arial", 14), width=25, relief="solid", bd=2)
user_input.pack(side="left", padx=10)

submit_button = ttk.Button(input_frame, text="Submit Data", command=process_input)
submit_button.pack(side="left")

# Log Table
log_frame = tk.Frame(main_frame)
log_frame.pack(fill="both", expand=True, pady=10)

columns = ("Timestamp", "Binary Input", "Status", "AI Suggestion")
log_table = ttk.Treeview(log_frame, columns=columns, show="headings", height=6)
for col in columns:
    log_table.heading(col, text=col, anchor="center")
    log_table.column(col, width=200, anchor="center")

log_table.pack(fill="both", expand=True)

# Export Button
export_button = ttk.Button(main_frame, text="Export Logs", command=export_logs)
export_button.pack(pady=10)

# Graph
fig, ax = plt.subplots(figsize=(6, 3))
canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().pack()

# Start reading data in a separate thread
threading.Thread(target=read_data, daemon=True).start()

# Run the application
root.mainloop()
