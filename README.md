# multivariate-linear-regression-l2
In this project, I implemented a **multivariate linear regression model from scratch**, including custom cost functions, gradient descent, and an optimizer with added support for **L2 Regularization (Ridge Regression)**. The goal was to predict startup profit based on R&D Spend, Administration, Marketing Spend, and State.

## Features
- Implemnentation Of Gradient Descent
- Cost Function with L2 Regularizatiion
- One-Hot encoding for categorical features
- Z-Score normalization of inputs/outputs
- Learning curve visualization
- Accuracy ecaluation using ±10% prediction margin
- Validated against 'scikit-learn''s 'LinearRegression'

---

## Dataset
The dataset is `50_Startups.csv`, a synthetic dataset of company budgets and profit from (Kaggle). Features include:
- R&D Spend
- Administration
- Marketing Spend
- State (California, New York, Florida)
- Profit (Target)

---

## What I learned 
- How to derive and implement the multivariate cost function and its gradient
- The impact of **L2 regularization** on model simplicity and overfitting
- How to interpret **learning curves** and training accuracy trade-offs
- How to build a machine learning pipeline from scratch without using libraries like `sklearn` or `statsmodels`

---

## Project Workflow
> I began by loading the dataset using `pandas`, exploring feature relationships with `seaborn.pairplot` for intuition. Since the `State` column was categorical, I manually implemented **one-hot encoding** (since it contained only 3 unique values). Next, I normalized all numerical features and the target (`Profit`) using **z-score normalization** to improve gradient descent stability.

> After preprocessing, I created the training matrices and implemented a **mean squared error (MSE)** cost function, followed by an MSE + L2 variant. I then manually computed the gradients for both versions — regularizing only the weights (`w`), not the bias (`b`), as is standard in ridge regression.

> I wrote two versions of gradient descent: one standard and one with L2 regularization. I experimented with different values of **learning rate (`alpha`)**, **regularization strength (`lambda`)**, and **iteration counts** to tune performance and convergence. By plotting the learning curves, I found my initial setup hadn't fully converged, so I increased iterations and adjusted the learning rate accordingly.

> After training, I compared the weights and bias values from my implementation to those produced by scikit-learn’s LinearRegression. The values were nearly identical, confirming the correctness of my implementation.

> Finally, I evaluated model accuracy using a **±10% prediction tolerance** and discovered that the L2-regularized model had ~2% lower training accuracy — indicating it was likely generalizing better by avoiding overfitting.

---
 
## Example Results

| Model Type         | Accuracy (±10%) | Weight Magnitude | Notes                         |
|--------------------|-----------------|------------------|-------------------------------|
| Unregularized      | 82%             | Larger           | Slightly higher training acc |
| L2-Regularized (λ=5)| 80%             | Smaller          | More general, simpler model  |

---

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Sklearn

