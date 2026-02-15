# Lending Club Analysis: A Beginner's Guide 🚀

Hi there! This project is all about analyzing data from **Lending Club**, which is a place where people lend money to each other. We want to look at their loan history from 2007 to 2017 to understand patterns, like who pays back their loans and who doesn't.

Think of this project like **preparing a big meal**. We have a huge pile of raw, messy ingredients (data), and we need to clean, chop, and organize them before we can cook something delicious (make charts and models).

---

## ⏱️ 10-Minute Quick Start Guide

Want to get this running on your computer right now? Follow these simple steps!

1.  **Get the Code**: Download this folder to your computer.
2.  **Install Python**: Make sure you have Python installed. (It's the tool that runs our scripts).
3.  **Open a Terminal**: This is where you type commands. Navigate to this folder.
4.  **Install Requirements** (The Grocery List):
    ```bash
    pip install -r requirements.txt
    ```
5.  **Step 1: The Delivery Truck (Load Data)**
    Run this command to get the raw data and unpack it:
    ```bash
    python src/data_loading.py
    ```
    *Wait a moment while it unzips the big file and organizes it.*

6.  **Step 2: The Chef (Clean Data)**
    Run this command to wash and chop the data:
    ```bash
    python src/data_cleaning.py
    ```
    *This will fix errors, fill in missing info, and make the data nice and tidy.*

7.  **Step 3: The Health Inspector (Verify Data)**
    Run this command to check if our work is good:
    ```bash
    python src/verify_cleaning.py
    ```
    *If it says "PASSED", you are ready to analyze!*

---

## 📂 File-by-File Explanation (ELI5)

Here is a breakdown of every file in this project, explained simply:

### 1. `src/data_loading.py` (The Delivery Truck 🚚)
- **What it is:** This script is responsible for getting the raw data into our project.
- **Why we need it:** The original data file is huge and sometimes zipped up. We can't just open it easily.
- **How it works:** 
    1. It checks if we have the raw CSV file.
    2. If not, it finds the ZIP file (`datasetzip.zip`) and extracts it automatically.
    3. It reads the giant file in small chunks (so your computer doesn't crash) and saves a more improved version (`optimized_accepted_data.csv`) that is easier to work with.

### 2. `src/data_cleaning.py` (The Chef 👨‍🍳)
- **What it is:** This script takes the raw data and cleans it up.
- **Why we need it:** Real-world data is messy! It has missing values, weird text (like "36 months" instead of just the number 36), and dates stored as text.
- **How it works:**
    - **Fixes Missing Values:** If `revol_util` is missing, it fills it with the average. If employment length is missing, it assumes 0 years.
    - **Standardizes Text:** It turns "36 months" into the number `36`.
    - **Fixes Dates:** It turns text dates (like "Dec-2015") into real calendar dates Python understands.
    - **Samples Data:** It creates a smaller, clean sample (`cleaned_sample_data.csv`) so we can test our analysis quickly without waiting for the huge file every time.

### 3. `src/verify_cleaning.py` (The Health Inspector 🕵️)
- **What it is:** This script checks the work of the Chef (`data_cleaning.py`).
- **Why we need it:** We need to trust our data. If the cleaner made a mistake, our analysis will be wrong.
- **How it works:** It runs a checklist:
    - "Is `term` a number?" ✅
    - "Are there any missing values in important columns?" ✅
    - "Are the dates real dates?" ✅
    - If anything fails, it yells at us so we can fix it!

### 4. `src/eda.py` (The Sketchpad ✏️)
- **What it is:** This is a placeholder file for future data exploration.
- **Why we need it:** We will use it later to draw charts and learn from the data.
- **How it works:** It's currently empty, waiting for your ideas!

### 5. `data/` (The Pantry 🥫)
- **What it is:** This is where we store our data files.
- **Why we need it:** To keep things organized.
    - `data/raw/`: The untouched, original ingredients.
    - `data/processed/`: The chopped, cleaned, and ready-to-use ingredients.

### 5. `requirements.txt` (The Grocery List 📝)
- **What it is:** A list of Python libraries (tools) we need.
- **Why we need it:** To make sure everyone has the same tools installed so the code runs exactly the same way on every computer.

---

## 🛠️ How to Re-create This Project

If you lost everything and only had the raw data `datasetzip.zip`, you could rebuild the entire analysis in **under 10 minutes** just by running the three scripts in order:

1.  `python src/data_loading.py`
2.  `python src/data_cleaning.py`
3.  `python src/verify_cleaning.py`

That's it! You're now a data pro! 🎉
