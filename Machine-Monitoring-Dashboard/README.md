# Machine-Monitoring-Dashboard


## Features
- Real-time machine status monitoring
- AI-based fault detection and repair suggestions
- Graphical data visualization
- Log export in CSV and Excel formats
- Standalone application support

## Repository URL
[GitHub Repository](https://github.com/shreyaspangavhane/Machine-Monitoring-Dashboard)

---

## Setup Instructions

### Prerequisites
Ensure you have the following installed:

#### Windows
- Python 3.x ([Download Python](https://www.python.org/downloads/))
- Pip package manager (comes with Python)
- Git (optional, for cloning the repository)

#### Linux (Ubuntu/Debian-based)
```bash
sudo apt update && sudo apt install python3 python3-pip git -y
```

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/shreyaspangavhane/Machine-Monitoring-Dashboard.git
cd Machine-Monitoring-Dashboard
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux
venv\Scripts\activate  # For Windows
```

### Step 3: Install Dependencies
```bash
pip install tkinter serial pandas matplotlib openai
```

### Step 4: Set OpenAI API Key
Create an environment variable:

#### Windows (Command Prompt)
```cmd
setx OPENAI_API_KEY "your-api-key"
```

#### Linux (Bash)
```bash
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Run the Application
```bash
python app.py
```

---

## Creating a Standalone Executable
To package the application as a standalone executable, use `pyinstaller`.

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Create an Executable
For Windows:
```cmd
pyinstaller --onefile --windowed app.py
```
For Linux:
```bash
pyinstaller --onefile --windowed app.py
```
The executable will be available in the `dist/` folder.

### Step 3: Running the Standalone Application
#### Windows
```cmd
dist\app.exe
```
#### Linux
```bash
./dist/app
```

---
Open app.exe file will be inside the dist folder.

## Exporting Logs
To export logs:
1. Click the "Export Logs" button.
2. Save as CSV or Excel file.

## Troubleshooting
- **Serial connection issues**: Ensure the correct COM port is set in `SERIAL_PORT` in `app.py`.
- **OpenAI API errors**: Check the API key and ensure an active internet connection.
- **Tkinter UI not opening**: Ensure Python includes Tkinter (`python -m tkinter`).


