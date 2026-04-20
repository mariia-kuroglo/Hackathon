TO RUN:
1. Prerequisites
Python: Ensure Python is installed. If not, download it from python.org.
Project Files: Extract the project files. All files must remain in the same root folder.

2. Install Dependencies
Open your terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux), navigate to the project folder using the cd command, and run: pip install pandas streamlit plotly
3. Launch the App
To start the AI Assistant, enter the following command: streamlit run Zeiss.py
Your default web browser will automatically open a new tab showing the application interface.


📂 Required Folder Structure

For the AI to "learn" the expert baseline and apply the dark theme correctly, your folder must look like this:

📄 Zeiss.py — The main application script.

📁 .streamlit/ — Contains config.toml (enables the Dark Neon theme).

📁 data/training/ — CRITICAL: Place all training files (S1_v4.csv to S10_v4.csv) here.



Methodology & Logic Summary
Our AI Assistant uses a Supervised Expert-Baseline approach to measure and optimize microscope energy efficiency. The logic follows three key steps:

1. Expert-Filtered Training (The "Gold Standard")
Instead of calculating simple averages from all training data, our model filters the datasets to find "perfect" efficiency. It only learns from segments where ZEISS experts labeled the recommended_action as "no_action". This ensures the AI knows exactly what a truly efficient workflow looks like.

2. Statistical Cleaning (IQR Method)
To prevent "dirty" data or extreme anomalies (like accidental 10-hour idle times) from skewing our reference, we applied the Interquartile Range (IQR) method. This removes statistical outliers, creating a "Clean Baseline" that represents the elite standard of microscope operation.

3. Precision Waste Calculation
The AI compares the uploaded workflow against this clean baseline across four dimensions:

Time & Energy: Identifying phases that exceed the expert-recommended duration.

Redundant Hardware: Calculating energy waste (in Watt-hours) when the Live-view camera is active during automated scans.

Optimization Alignment: Mapping anomalies to the ZEISS 9R Strategy (e.g., R1 Rethink, R2 Reduce) to provide actionable, data-driven recommendations.

In short: We don't just compare files; we compare your performance against a mathematically purified model of expert efficiency, allowing us to pinpoint the exact Watt-hours wasted in every session.
