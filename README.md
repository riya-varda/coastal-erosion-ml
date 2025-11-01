# Coastal Erosion and Land Submergence Monitoring using Machine Learning

## Overview
This project focuses on analyzing and predicting land submergence and coastal erosion using machine learning models.  
It combines numerical datasets from Kaggle with satellite imagery from the Copernicus platform to understand how sea-level rise and temperature variations affect coastal regions.

The objective is to build predictive models that can identify regions at risk of submergence and support climate resilience planning.

---

## Project Workflow

### Phase 1: Data Collection
- Collected a land submergence dataset from Kaggle containing variables such as Year, Country, Land Area, Coastal Length, Temperature, Sea Level, and Land Impact.
- Collected satellite imagery data from the Copernicus platform to validate patterns and to explore future NDWI (Normalized Difference Water Index) analysis.

### Phase 2: Data Preprocessing
- Cleaned missing values and handled null entries in temperature and annual balance fields.
- Encoded categorical columns such as Country and Continent.
- Normalized and scaled numeric values to ensure consistency for model input.

### Phase 3: Random Forest Model
- Implemented a Random Forest Regressor to predict the extent of land impact due to submergence.
- Trained and tested the model on an 80/20 data split.
- Evaluated the model using R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

### Phase 4: XGBoost Model
- Built and tuned an XGBoost Regressor for comparison with the Random Forest model.
- Analyzed feature importance and the correlation between different environmental and geographical factors.
- Visualized the results using scatter plots and feature importance charts.

### Phase 5: Visualization
- Created a correlation heatmap to show relationships between sea level, temperature, and land impact.
- Visualized model performance through Actual vs Predicted plots.
- Compared model metrics side by side for easier interpretation.

### Phase 6: Integration and Export
- Compiled all model outputs, comparison tables, and prediction results.
- Exported evaluation metrics and predictions as CSV files for external analysis or integration into a dashboard.
- Documented future research directions and applications.

---

## Results Summary

| Model | R² Score | MAE | RMSE |
|-------|-----------|-----|------|
| Random Forest | ~0.80 | ~0.30 | ~0.25 |
| XGBoost | ~0.87 | ~0.22 | ~0.19 |

The XGBoost model demonstrated higher accuracy and lower error rates compared to the Random Forest model, making it a better choice for predicting land submergence levels.

---

## Challenges Faced
- Copernicus satellite image files were large in size (often exceeding 1 GB), which caused slow processing and limited Colab memory.
- Aligning and preprocessing satellite imagery for NDWI analysis requires specialized geospatial tools.
- Missing or incomplete data for certain countries introduced noise into the training process.
- Limited availability of long-term coastal monitoring data in some regions made temporal modeling more difficult.

---

## Future Work
- Implement NDWI (Normalized Difference Water Index) to analyze changes in water coverage using satellite imagery.
- Use deep learning techniques such as U-Net for land–water segmentation in Copernicus imagery.
- Integrate real-time satellite data pipelines to update coastal erosion risk levels dynamically.
- Develop a Streamlit or Dash web application for interactive visualization and forecasting.

---

## Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- Google Colab for training and experimentation
- Copernicus Open Access Hub for satellite imagery
- Kaggle for open environmental datasets

---

## Repository Contents

| File | Description |
|------|--------------|
| `land_submergence.csv` | Primary dataset used for model training |
| `Coastal_Erosion_Project.ipynb` | Main Jupyter/Colab notebook containing the entire workflow |
| `model_comparison_results.csv` | Comparison of Random Forest and XGBoost model metrics |
| `predictions_output.csv` | Actual vs Predicted land impact data |
| `README.md` | Project documentation and report |

---

## Author
**Riya Varda**  
B.E. Engineering, MIT Bangalore  
Focused on applying data science and AI to solve environmental and sustainability challenges.
