# EduPro Demand & Revenue Forecast

**EduPro Forecast** is a predictive machine learning system and interactive Streamlit dashboard built to optimize course launches. It accurately forecasts future course enrollments and revenue based on historical data, instructor profiles, and course metadata.

**[Read the Full Research Paper](./RESEARCH_PAPER.md)**

---

## Features
- **Real-Time Forecasting:** Instantly predicts enrollment and revenue using a custom Random Forest Regressor ($R^2 > 0.90$).
- **Unit Economics:** Calculates simulated Revenue Per Enrollment (RPE) to optimize pricing strategies.
- **Feature Importance Analysis:** Identifies the key drivers behind course success (e.g., category, instructor experience).
- **Clean UI:** Professional, responsive dashboard built with Streamlit and styled with custom CSS & Google Material Symbols.

## Application Demo
<video src="./streamlit-app-2026-09-30-01-54-14.webm" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

## Tech Stack
- **Python 3**
- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Dashboard UI:** `streamlit`, `plotly`
- **Deployment:** Streamlit Community Cloud

## Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohehet/edupro-forecast.git
   cd edupro-forecast
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

---
*Built by the EduPro Data Science Team*
