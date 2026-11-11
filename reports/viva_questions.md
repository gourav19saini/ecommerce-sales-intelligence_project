# Viva Preparation — Likely Questions & Answers

## Python

**1. Why Python for this project?**
Python has mature, well-documented libraries for every stage of the
pipeline used here — Pandas for data manipulation, Matplotlib/Seaborn for
visualization, and scikit-learn for machine learning — plus a syntax that
keeps the code readable for a student explaining it later.

**2. List vs tuple?**
A list is mutable (can be changed after creation) and written with `[]`;
a tuple is immutable and written with `()`. Tuples are slightly faster and
are used when the data shouldn't change, e.g. a fixed set of feature-column
names.

**3. Function vs lambda?**
A `def` function can hold multiple statements, docstrings, and a name for
reuse and readability (used throughout `src/`); a `lambda` is a small,
unnamed, single-expression function, useful for quick inline operations
like a custom sort key.

**4. How is exception handling used or relevant here?**
While the core pipeline here is deterministic on a clean dataset, robust
data-loading code should wrap file reads (e.g. `pd.read_csv`) in
`try/except` to handle missing files or encoding issues gracefully rather
than crashing.

## Pandas

**5. What is a DataFrame?**
A 2-dimensional, labeled data structure in Pandas — like a spreadsheet or
SQL table — with rows and named columns of potentially different types.
It is the core data structure used throughout this project.

**6. Why use groupby?**
`groupby` splits the data into groups based on one or more columns (e.g.
`Category`), applies an aggregation (e.g. `sum` of `Sales`), and combines
the results — this is how revenue-by-category, revenue-by-region, and
per-customer/per-product features were all computed.

**7. How did you handle missing values?**
This particular dataset had none (verified dynamically with
`df.isnull().sum()`), but the cleaning pipeline is written to handle them
if present: numeric columns are filled with the median (robust to
outliers), categorical columns with the literal string `'Unknown'`, and
rows missing the order date are dropped since a transaction date cannot be
reliably imputed.

**8. How did you detect duplicates?**
With `df.duplicated().sum()` to count exact duplicate rows, and
`df.drop_duplicates()` to remove only genuine full-row duplicates — no
partial-match "fuzzy" duplicates were removed.

## EDA

**9. What is EDA?**
Exploratory Data Analysis — using summary statistics and visualizations to
understand a dataset's structure, patterns, and relationships before
modeling.

**10. Why visualize data?**
Charts reveal patterns (trends, seasonality, outliers, relationships)
that are hard to see from raw numbers alone, and communicate findings to a
non-technical audience.

**11. What is correlation?**
A statistical measure (ranging from -1 to +1) of the strength and
direction of the *linear* relationship between two numerical variables.

**12. Why doesn't correlation imply causation?**
Two variables can move together because of a third, unmeasured factor, or
by coincidence, without either one causing the other. For example, higher
discounts and lower profit are correlated here, but that alone doesn't
prove discounting is the sole cause of a profit drop — product mix, season,
and category also vary alongside discount level.

## Machine Learning

**13. What is supervised learning?**
Training a model on input features paired with known target values (here,
monthly sales) so it learns a mapping it can apply to new, unseen inputs.

**14. Why regression (not classification)?**
The target — monthly sales revenue — is a continuous numeric value, not a
category, so regression is the appropriate supervised-learning task.

**15. What is train-test split?**
Dividing the data into a training portion (used to fit the model) and a
test portion (used only to evaluate it), so performance is measured on
data the model has never seen.

**16. Why chronological splitting here specifically?**
Because this is a time-series forecasting problem, a random shuffle would
let the model train on future months and be tested on earlier ones,
artificially inflating performance. The chronological split (first 80% of
months train, last 20% test) mirrors how the model would actually be used
in practice — predicting the future from the past.

**17. What is overfitting?**
When a model learns the noise/specifics of the training data too closely
and performs much worse on new data. In this project, the tree-based
models overfit the training months given how few (36) were available.

**18. What is underfitting?**
When a model is too simple to capture the real pattern in the data, so it
performs poorly on both training and test data.

**19. What is feature engineering?**
Creating new input variables from raw data that make the underlying
pattern easier for a model to learn — e.g. turning a raw date into
year/month/quarter, or turning transaction history into lag and
rolling-average sales features.

## Metrics

**20. What is MAE?**
Mean Absolute Error — the average absolute difference between predicted
and actual values, in the same units as the target (dollars here). Easy to
interpret; treats all errors equally.

**21. What is RMSE?**
Root Mean Squared Error — like MAE, but squares errors before averaging
(then takes the square root), so larger errors are penalized more heavily.

**22. What is R²?**
The proportion of variance in the target that the model explains, from 0
(no better than predicting the mean) to 1 (perfect fit). This project's
selected model achieved an R² of 0.9373 on the held-out test months.

**23. Which metric did you use and why?**
All three (MAE, RMSE, R²) together — MAE gives an intuitive dollar-value
error, RMSE highlights whether any months were badly missed, and R² gives
a normalized sense of overall fit. The final model was selected by lowest
RMSE.

## Project

**24. Why did you select this problem?**
It combines every stage of the Data Science/AI-ML training curriculum —
cleaning, EDA, feature engineering, regression, and evaluation — in one
realistic business context (retail sales forecasting).

**25. What was the biggest data-cleaning challenge?**
This dataset happened to be already clean (no missing values or
duplicates), so the main preprocessing work was type conversion and a
careful, non-destructive outlier analysis — deciding that extreme
Sales/Profit values were legitimate rather than deleting them by default.

**26. Which features were most important?**
Based on the Random Forest's feature importances, `Monthly_Quantity` and
`Monthly_Orders` were by far the most influential — months with more
orders and higher volume directly produce more revenue.

**27. Why did one model outperform another?**
Linear Regression outperformed the tree-based models because there were
only 36 monthly training observations — too few for the more flexible
tree-based models to find stable, generalizable split points, so they
overfit the training months. The lag/rolling features already capture
most of the month-to-month sales momentum in a roughly linear way, which
favors a linear model.

**28. What are the limitations of this project?**
A small number of monthly observations (45 total), a single retailer's
data from 2014–2017, and forecasting features that assume recent
seasonality continues rather than anticipating structural change.

**29. How would you improve the project?**
Use weekly rather than monthly granularity if more historical data became
available, add external drivers (marketing spend, holiday calendars), and
compare against dedicated time-series models like SARIMA or Prophet.

**30. How is data leakage avoided in your forecasting features?**
Lag and rolling-average features are built using `.shift()` before any
rolling computation, so every month's feature row only uses information
from strictly earlier months — never the current or a future month's
sales.
