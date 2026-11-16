# Salifort Motors Employee Churn Analysis

Predicting which employees are likely to leave Salifort Motors, and why, using Python, logistic regression, and tree-based machine learning models, with retention recommendations for HR and senior leadership.

**Author:** Muhammad Zeeshan Siddiqui · [GitHub](https://github.com/ZEESHANSIDDIQUIgit)

---

## Business Problem

Salifort Motors has a high employee turnover rate, which is costly in recruiting, training, and upskilling. HR surveyed a sample of employees and asked for two things:

1. A model that predicts whether an employee will leave.
2. Data-driven insights into what drives turnover, with recommendations to improve retention.

**Target variable:** `left` (1 = left the company, 0 = stayed)

## Dataset

[HR Analytics and Job Prediction (Kaggle)](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv): 14,999 rows and 10 columns.

| Column | Description |
|---|---|
| `satisfaction_level` | Employee-reported job satisfaction (0-1) |
| `last_evaluation` | Score of last performance review (0-1) |
| `number_project` | Number of projects the employee contributes to |
| `average_monthly_hours` | Average hours worked per month |
| `tenure` | Years at the company |
| `work_accident` | Whether the employee had a workplace accident |
| `promotion_last_5years` | Whether the employee was promoted in the last 5 years |
| `department` | Employee's department |
| `salary` | Salary band (low, medium, high) |
| `left` | Whether the employee left (target) |

## Approach (PACE framework)

1. **Plan:** defined the stakeholders (HR and senior leadership) and the prediction task (binary classification).
2. **Analyze:**
   - Renamed columns to consistent `snake_case` and fixed misspellings.
   - Found no missing values and removed 3,008 duplicate rows (20% of the data), leaving 11,991 rows.
   - Identified 824 tenure outliers (removed only for logistic regression).
   - About 16.6% of employees in the cleaned data left.
3. **Construct:**
   - **Logistic regression:** encoded `salary` as ordinal and `department` with dummies, then used a stratified train/test split.
   - **Decision tree and random forest:** tuned with `GridSearchCV` (4-fold cross-validation, scored on accuracy, precision, recall, F1, and AUC).
   - **Feature engineering (round 2):** dropped `satisfaction_level` (it may not be available for every employee in practice) and replaced `average_monthly_hours` with a binary `overworked` flag (more than 175 hours per month) to reduce data leakage.
4. **Execute:** evaluated the champion on a held-out test set, extracted feature importances, and wrote recommendations.

## Results

**Logistic regression (test set)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Predicted would leave | 0.44 | 0.26 | 0.33 |
| Weighted average | 0.79 | 0.82 | 0.80 |

Accuracy was 82%. Logistic regression struggled to catch the employees who actually leave, which is the group that matters most.

**Tree-based models, round 2 (without `satisfaction_level`), cross-validation**

| Model | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| Decision tree | 0.857 | 0.904 | 0.879 | 0.959 | 0.959 |
| Random forest | 0.867 | 0.879 | 0.872 | 0.957 | 0.965 |

**Champion: random forest, round 2 (test set)**

| Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|
| 0.870 | 0.904 | 0.887 | 0.962 | 0.938 |

The round 1 random forest (all features, including satisfaction) scored higher on the test set (F1 0.941, accuracy 0.981), but those scores are likely inflated by data leakage, so the round 2 model is the more realistic one.

**Top predictors (both tree models):** `last_evaluation`, `number_project`, `tenure`, `overworked`

## Key Insights

- Employees who left fall into two groups: **overworked** employees (many projects, very long hours, near-zero satisfaction) and **under-engaged** employees (fewer hours, lower evaluations, moderate satisfaction).
- Every employee with 7 projects left.
- Four-year employees who left had unusually low satisfaction.
- Employees with more than six years of tenure rarely leave.
- Very few promoted employees left, and very few of the longest-hours employees were promoted.
- Department and salary showed little difference in who leaves.

## Recommendations

- Cap the number of projects per employee.
- Investigate why four-year employees are so dissatisfied, and consider promoting those who have reached that tenure.
- Either reward employees for working longer hours or stop requiring them to.
- Make overtime pay and time-off policies clear.
- Hold company-wide and team-level discussions about workload and work culture.
- Don't reserve high evaluation scores for employees who work 200+ hours per month; reward effort fairly across the scoring scale.

## Limitations and Next Steps

- The data comes from a single survey sample, so results may not generalize.
- High scores in round 1 likely reflect leakage, so deployment should rely on the round 2 approach.
- The model produces more false positives than false negatives, so some employees will be flagged as at risk when they are not.
- Next steps: validate on newer data, try XGBoost, and test a version without `last_evaluation`.

## Repository Structure

```
salifort-motors-employee-churn-analysis/
├── Salifort_Motors_project.ipynb   # Full analysis notebook
├── models/                         # hr_rf1.pickle, hr_rf2.pickle
├── executive_summary.md            # One-page summary for stakeholders
└── README.md
```

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · Jupyter Notebook

## How to Run

```bash
git clone https://github.com/ZEESHANSIDDIQUIgit/salifort-motors-employee-churn-analysis.git
cd salifort-motors-employee-churn-analysis
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook Salifort_Motors_project.ipynb
```

> **Note:** download the dataset from Kaggle and place it next to the notebook as `HR_capstone_dataset.csv`. The saved `.pickle` models should be loaded with the same scikit-learn version used to create them, and only load pickle files you trust.
