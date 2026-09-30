# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

**Authors:** Mohit Kumar
**Date:** September 2026  

---

## Abstract
For online learning platforms like EduPro, future demand visibility is critical. Decisions regarding launching new courses, adjusting pricing, and onboarding instructors have historically relied on intuition rather than empirical evidence. This paper presents the development and deployment of an end-to-end machine learning system designed to transition EduPro from reactive reporting to proactive planning. By aggregating transactional, teacher, and course metadata, we engineered robust predictive models capable of forecasting course enrollment and revenue with high accuracy (R² > 0.90). The resulting intelligence is deployed via a real-time Streamlit dashboard, enabling stakeholders to simulate unit economics before a course is launched.

---

## 1. Introduction

### 1.1 Problem Statement
EduPro currently lacks quantitative evidence to support strategic decisions such as course launches and pricing optimization. The absence of predictive models for enrollment demand and revenue forecasting at the course and category levels leads to inefficiencies and missed revenue opportunities. 

### 1.2 Objective
The primary objective of this project was to construct predictive intelligence that answers a central question: *Which future courses will attract enrollments and generate the most revenue?* We aimed to build and deploy two distinct predictive models (Enrollment and Revenue) and expose them through an interactive, open-source dashboard.

---

## 2. Methodology

### 2.1 Data Architecture and Preprocessing
The foundational dataset consisted of three primary relational schemas:
1. **Courses:** Metadata including price, duration, level, and category.
2. **Teachers:** Instructor profiles, experience years, and aggregate ratings.
3. **Transactions:** 10,000 individual historical transaction records.

The data was transformed from a transactional level to an aggregated *course-level* dataset ($N = 887$). We calculated `enrollment_count` and `course_revenue` (enrollment × price) for each unique course.

### 2.2 Feature Engineering
To capture the nuances of user purchasing behavior, several categorical and ordinal features were engineered:
- **Price Bands:** Categorized into Low (≤$20), Medium ($21-$100), and High (>$100).
- **Duration Buckets:** Segmented into Short (≤10h), Medium, Long, and Very Long.
- **Experience & Rating Tiers:** Grouped instructors into Junior, Mid, Senior, and Expert levels, alongside categorizing course ratings.
- **Match Score:** A derived metric evaluating the alignment between an instructor's expertise and the course category.

### 2.3 Model Selection and Pipeline Design
We evaluated baseline linear models (Ridge Regression) against complex non-linear ensemble methods (Random Forest, Gradient Boosting). 

The chosen architecture utilized a `scikit-learn` Pipeline ensuring zero data leakage:
- **Numerical Processing:** Imputation of missing values (median) followed by Standardization (`StandardScaler`).
- **Categorical Processing:** One-Hot Encoding (`OneHotEncoder`) handling unknown categories gracefully.
- **Estimator:** `RandomForestRegressor` with optimized hyperparameters (`n_estimators=100`, `random_state=42`).

*Crucial Architecture Note:* We instantiated distinct, independent pipelines for Enrollment and Revenue to prevent object-reference contamination, ensuring that the preprocessing state of one target did not overwrite the other.

---

## 3. Results and Evaluation

### 3.1 Model Performance
The Random Forest models vastly outperformed the linear baselines, successfully capturing the non-linear interactions between instructor ratings, price, and category demand.

| Target Model | R² Score | Mean Absolute Error (MAE) | 
| :--- | :--- | :--- |
| **Enrollment Model** | 0.95 | ± 12 enrollments |
| **Revenue Model** | 0.91 | ± $850 |

An R² of 0.95 indicates that 95% of the variance in future course enrollments can be explained by our engineered feature set.

### 3.2 Feature Importance Analysis
Analysis of the Gini impurity reduction within the Random Forest revealed the primary drivers of demand:
1. **Category:** The subject matter remains the dominant factor in both enrollment and revenue generation.
2. **Course Level:** Beginner courses demonstrated a statistically significant higher enrollment volume, though Advanced courses yielded higher Revenue Per Enrollment (RPE).
3. **Instructor Experience:** Highly correlated with sustained revenue across long-duration courses.

---

## 4. Deployment and Application

### 4.1 Application Demo
Below is a demonstration of the real-time forecasting application in action:

<video src="./streamlit-app-2026-09-30-01-54-14.webm" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

### 4.2 System Architecture

The models were serialized using `joblib` and deployed into a production Streamlit environment. The UI was designed utilizing a custom CSS framework integrating Google Material Symbols to provide a rich, professional user experience without heavy front-end frameworks.

**Key Dashboard Features:**
- **Real-Time Inference:** Stakeholders can adjust sliders (e.g., changing course price from $50 to $75) to instantly see the predicted impact on total revenue and enrollment volume.
- **Unit Economics:** Calculates simulated Revenue Per Enrollment (RPE).
- **Global Accessibility:** Hosted continuously on Streamlit Community Cloud, linked directly via continuous integration with GitHub.

---

## 5. Conclusion
The implementation of this machine learning pipeline successfully transitions EduPro into a data-driven organization. By accurately forecasting course viability *before* resources are expended on production, EduPro can optimize its catalog, maximize instructor ROI, and dynamically adjust pricing structures to meet market demand.

---

