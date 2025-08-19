# tree-pruning
 Interactive decision-tree pruning playground (pre/post pruning) with SQL feature engineering in DuckDB, Plotly widgets, and a shareable Streamlit app.


Pruning Playground 🌳 — Interactive Decision Trees with SQL + Streamlit

This project is a hands-on lab for understanding binary decision trees and pruning. You’ll build a classifier on the Adult Income dataset, engineer features with SQL (DuckDB), and compare pre-pruning (e.g., max_depth, min_samples_leaf) with post-pruning (ccp_alpha). Everything is interactive: sliders update ROC, confusion matrix, and metrics live in both the Jupyter notebook and a Streamlit web app.


## Key features

Binary decision tree classifier (scikit-learn).

Pre-pruning: tune max_depth, min_samples_leaf, min_samples_split.

Post-pruning: explore cost-complexity pruning with ccp_alpha.

SQL EDA & feature engineering using DuckDB directly on pandas DataFrames.

Interactive visuals: Plotly ROC, confusion matrix with threshold slider, feature importance.

Streamlit app for a shareable, clickable demo.


## Tech stack

Python 3.11, pandas, numpy, scikit-learn

DuckDB (SQL inside your notebook)

Plotly, ipywidgets (Jupyter interactivity)

Streamlit (deployable web UI)
