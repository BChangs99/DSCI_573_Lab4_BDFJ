# NYC Airbnb 2019: End-to-End Regression Project

**Authors**: Yu Yuan (Brian) Chang, Farhan Bin Faisal, Daria Khon, Adrian Leung, Zhiwei Zhang

---

## Project Overview

This repository hosts a regression project aimed at predicting the popularity of New York City Airbnb listings (as measured by **reviews per month**) using data from the [NYC Airbnb Open Data (2019)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data). By analyzing features such as location, minimum nights, availability, and host attributes, we seek to provide insights that help Airbnb hosts optimize their listings.

### Motivation

- **Hosts** can better tailor their listings to increase engagement and bookings.  
- **Travelers** benefit from more accurate representation of popular neighborhoods and room types.  
- **Data enthusiasts** can explore additional feature engineering or modeling techniques to enhance predictive performance.

---

## Data

- **Source**: [Kaggle – New York City Airbnb Open Data (2019)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)  
- **Features**: Includes listing location (latitude, longitude, neighborhood group), property attributes (room type, price), availability, and textual descriptions (e.g., listing names).  
- **Target**: `reviews_per_month` – serves as a proxy metric for listing popularity.

### Preprocessing & Feature Engineering

- Handled missing values for `reviews_per_month` (tied to zero `number_of_reviews`).  
- Extracted month from the `last_review` date to capture seasonal patterns.  
- Performed VADER sentiment analysis on the `name` feature to gauge the listing’s textual sentiment.

---

## Models

A variety of regression models were trained and evaluated:

1. **Baseline**  
   - *DummyRegressor*: Predicts the mean of the target; used as a simple reference.

2. **Linear Models**  
   - *Ridge / Elastic Net*: Simple and interpretable, with built-in regularization.

3. **Ensemble Methods**  
   - *Random Forest Regressor*: Good for capturing non-linearities but can overfit.  
   - *LightGBM (LGBM) Regressor*: Gradient boosting framework known for efficiency and strong performance.

### Hyperparameter Tuning & Evaluation

- **Cross-Validation**: Used to assess model generalizability.  
- **RandomizedSearchCV**: Explored key hyperparameters such as `max_depth`, `learning_rate`, and regularization coefficients.  
- **Performance Metrics**: 
  - *R² (coefficient of determination)* for explanatory power.  
  - *Mean Absolute Error (MAE)* or *Mean Squared Error (MSE)* for accuracy in predictions.

### Key Results

- **Best Model**: LightGBM (LGBM) with parameters *(learning_rate=0.05, max_depth=50, n_estimators=200)*.  
- **Cross-Validation R²**: ~0.66  
- **Test Set R²**: ~0.69  
- The small gap between CV and test R² indicates the model generalizes well.

---

## Repository Structure

```plaintext
.
├── data/
│   └── raw/               # Raw CSV or downloaded data
├── analysis/
│   └── report.ipynb     # Main exploratory and modeling notebook
├── results/
|   ├── figures/
│   ├── models/lgbm_random.pickle # Example saved model objects
│   └── tables/optimized_results.csv # Summary of hyperparameter tuning scores
├── src/
│   |── eda.py   # Utility functions
|   |── ...
├── environment.yml           # Conda environment specification
├── LICENSE                   # License file
└── README.md                 # This file


# Usage
## Running the analysis

1. Clone this repository `git clone`
2. Set up the environment
    ```
    conda env create -f environment.yml
    conda activate nyc_airbnb_env
    ```
3. Enter the root project direcory on your local repository and run:  
   ```
   make quarto
   ```


# License
This project was created with the [`MIT License`](LICENSE.md)

# Acknowledgements
- Dataset: Provided by Kaggle.
- Contributors: Yu Yuan (Brian) Chang, Farhan Bin Faisal, Daria Khon, Adrian Leung, Zhiwei Zhang.
- Tools & Libraries: Python 3, pandas, scikit-learn, LightGBM, Altair, etc.