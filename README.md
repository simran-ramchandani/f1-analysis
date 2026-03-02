# f1-analysis
F1 Race Analytics Dashboard

A Python-based Formula 1 analytics project that uses FastF1 data to analyze race sessions, driver performance, lap data, and telemetry.

This project fetches official F1 session data, caches it locally, and presents insights through a simple web interface.

## What This Project Does

Loads race and qualifying session data

Uses cached FastF1 data for faster performance

Extracts:

Lap times

Position data

Weather data

Driver information

Session status

Displays processed results through an HTML template

Basically:
Raw F1 timing data → structured analytics → simple frontend display.

🛠️ Tech Stack

Python

FastF1

Flask (for web interface)

HTML templates

SQLite cache (via FastF1)

📂 Project Structure
proj/
│
├── project.py              # Main application logic
├── templates/
│   └── index.html          # Frontend page
│
├── cache/                  # FastF1 session cache
│   ├── 2024/
│   ├── 2025/
│   └── 2026/
What the cache folder means

FastF1 downloads official session data once and stores it locally as .ff1pkl files.

Example:

2025-05-25_Monaco_Grand_Prix/
    └── 2025-05-25_Race/
        ├── car_data.ff1pkl
        ├── weather_data.ff1pkl
        ├── lap_count.ff1pkl

So next time you run the same session → it loads instantly.

⚙️ How to Run the Project
1. Install dependencies
pip install fastf1 flask pandas matplotlib
2. Run the application
python project.py
3. Open in browser

Go to:
http://127.0.0.1:5000
