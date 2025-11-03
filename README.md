# AL-ML-Learning-Tutorials
This Repo will have python code to understand to Machine learning and AI related topics

## 🧠 What You’ll Learn

- Python Data Structures (Lists, Dictionaries, Sets, Tuples)  
- Writing and Using Functions (Regular, Lambda, Map, Filter)  
- Working with Modules and Packages  
- File Handling and Path Management  
- Exception Handling and Program Flow Control  
- Object-Oriented Programming (Classes and Objects)  
- Advanced Python Concepts (Decorators, Generators, Context Managers)  
- Data Analysis and Feature Engineering  
- Real-world Assignments and Mini Projects  
- Tensor Operations with NumPy/PyTorch  

---

## 🧰 Prerequisites

- **Python 3.x** installed on your system  
- **Jupyter Notebook** or **VS Code** with Python support  

---

## 📘 License

This repository is intended **for educational purposes only**.  
Feel free to explore, experiment, and adapt the examples to enhance your learning.

---

### 🌟 Explore | Learn | Practice | Master Python 🚀

## Quick setup — run the notebooks locally (beginner-friendly)

Follow these minimal steps on Windows using PowerShell. These assume you have Python 3.8+ installed. If you don't, install Python from https://www.python.org/downloads/ and be sure to check "Add Python to PATH" during installation.

1) Clone this repository

```powershell
git clone https://github.com/yuvaraj-ai-dev/AL-ML-Learning-Tutorials.git
cd AL-ML-Learning-Tutorials
```

2) Create and activate a virtual environment (recommended)

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# If you see a policy error when activating, run this once as Administrator OR in your user session:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then re-run: .\.venv\Scripts\Activate.ps1
```

3) Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt

# If the repository has no requirements or you get an error, install Jupyter manually:
pip install notebook
```

4) Start Jupyter Notebook (open in your browser)

```powershell
# Start the notebook server and open the browser
python -m notebook

# Alternatively open a single notebook directly (it will still run the notebook server):
python -m notebook "Python-Basics/1.Fundamentals.ipynb"
```

5) Using VS Code instead of Jupyter Notebook (optional)

- Open VS Code and select "Open Folder..." and choose the cloned `AL-ML-Learning-Tutorials` folder.
- Install the official "Python" extension by Microsoft if you haven't already.
- Select the Python interpreter from the bottom-right (choose `.venv\Scripts\python.exe`).
- Open `Python-Basics/1.Fundamentals.ipynb` in VS Code — you can run cells inline using the notebook UI.

6) Quick verification

```powershell
# With the virtual environment active, run a small check:
python -c "print('Python OK:', __import__('sys').version)"
python -c "import notebook; print('notebook package found')"
```

Troubleshooting tips
- If `python` isn't recognized, try `py` instead of `python` (e.g. `py -3 -m venv .venv`).
- If activation fails due to execution policy, run the PowerShell command shown above (RemoteSigned) for the CurrentUser scope.
- If a package fails to install, copy the exact pip error and search the message; often missing build-tools or a typo in `requirements.txt` is the cause.

Next steps
- Open `Python-Basics/1.Fundamentals.ipynb` and run the cells in order. The notebook contains beginner-friendly explanations and runnable examples about variables.
- If you'd like, I can create a short `variables_examples.py` script with the same examples so you can run them directly in PowerShell.

If you want me to run the notebook examples here and capture outputs, say "run examples" and I'll execute and report results.
