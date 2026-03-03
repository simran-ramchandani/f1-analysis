# F1 Race Analytics Dashboard

A Python-based Formula 1 analytics project that uses FastF1 data to
analyze race sessions, driver performance, lap data, and telemetry.

This project fetches official F1 session data, caches it locally, and
presents insights through a simple web interface.

------------------------------------------------------------------------

## What This Project Does

-   Loads race and qualifying session data\
-   Uses cached FastF1 data for faster performance\
-   Extracts:
    -   Lap times
    -   Position data
    -   Weather data
    -   Driver information
    -   Session status
-   Displays processed results through an HTML template

Raw F1 timing data to structured analytics to frontend display.

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   FastF1\
-   Flask\
-   HTML templates\
-   SQLite cache (via FastF1)

------------------------------------------------------------------------

## Project Structure

proj/  
├── project.py Main application logic\
├── templates/\
│ └── index.html Frontend page\
│\
├── cache/ FastF1 session cache\
│ ├── 2024/\
│ ├── 2025/\
│ └── 2026/

------------------------------------------------------------------------

## How to Run

1.  Install dependencies:

    pip install fastf1 flask pandas matplotlib

2.  Run the application:

    python project.py

3.  Open your browser:

    http://127.0.0.1:5000

------------------------------------------------------------------------

## Example Use Case

-   Compare driver lap times\
-   Analyze weather impact\
-   Track position changes\
-   Study race pace consistency


