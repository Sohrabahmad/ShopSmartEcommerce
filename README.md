# 🛍️ ShopSmart — Online Shopping Purchase Prediction

## 📌 Project Overview

**ShopSmart** is a machine learning classification project that predicts whether an online shopping session will result in a purchase.

The project uses a **Decision Tree Classifier** to analyze visitor behavior, browsing activity, page values, traffic sources, and other session characteristics to predict purchase conversion.

It includes a preprocessing pipeline, class imbalance handling, model evaluation, and hyperparameter tuning using `GridSearchCV`.

## 🎯 Objectives

- Analyze online shopping session data.
- Predict purchase conversion using machine learning.
- Preprocess numerical and categorical features.
- Handle class imbalance using balanced class weights.
- Evaluate model performance using F1 score and classification metrics.
- Tune decision tree hyperparameters using cross-validation.
- Build a reusable Scikit-learn pipeline.

## 📂 Dataset Description

The project uses `shop_smart_ecommerce.csv`.

- **Total records:** 12,330
- **Total columns:** 18
- **Input features:** 17
- **Target variable:** `Revenue`

### Dataset Features

| Feature | Description |
|---|---|
| `Administrative` | Number of administrative pages visited |
| `Administrative_Duration` | Time spent on administrative pages |
| `Informational` | Number of informational pages visited |
| `Informational_Duration` | Time spent on informational pages |
| `ProductRelated` | Number of product-related pages visited |
| `ProductRelated_Duration` | Time spent on product-related pages |
| `BounceRates` | Bounce rate for the session |
| `ExitRates` | Exit rate for pages visited |
| `PageValues` | Estimated page value |
| `SpecialDay` | Closeness of the visit to a special day |
| `Month` | Month of the session |
| `OperatingSystems` | Operating system category |
| `Browser` | Browser category |
| `Region` | Visitor region |
| `TrafficType` | Traffic source category |
| `VisitorType` | Type of visitor |
| `Weekend` | Whether the visit occurred on a weekend |
| `Revenue` | Target indicating whether a purchase occurred |

### Target Distribution

| Target | Records |
|---|---:|
| `False` — No purchase | 10,422 |
| `True` — Purchase | 1,908 |

The dataset is imbalanced, with fewer purchase sessions than non-purchase sessions. The model uses balanced class weights to help address this imbalance.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

1. Load the online shopping dataset.
2. Separate input features and the target variable.
3. Identify numerical and categorical columns.
4. Split the data into training and testing sets.
5. Standardize numerical features.
6. One-hot encode categorical features.
7. Train a Decision Tree Classifier.
8. Evaluate predictions using F1 score, classification report, and confusion matrix.
9. Tune model hyperparameters using `GridSearchCV`.
10. Identify the best configuration using five-fold cross-validation.

## 🧹 Data Preprocessing

### 1. Feature and Target Separation

```python
X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)
```

### 2. Train-Test Split

The dataset is divided into training and testing sets using stratified sampling.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratification preserves the target-class proportions.

### 3. Feature Preprocessing

The project uses `ColumnTransformer` to apply separate preprocessing operations to numerical and categorical features.

- **Numerical features:** `StandardScaler`
- **Categorical features:** `OneHotEncoder(handle_unknown="ignore")`

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)
```

## 🤖 Machine Learning Model

### Decision Tree Classifier

The initial model uses the following configuration:

```python
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```

| Parameter | Value | Purpose |
|---|---:|---|
| `max_depth` | 6 | Limits tree depth |
| `min_samples_leaf` | 30 | Requires a minimum number of samples per leaf |
| `class_weight` | `balanced` | Adjusts class weights according to class frequencies |
| `random_state` | 42 | Supports reproducibility |

### Pipeline Implementation

The preprocessing and model steps are combined into a single Scikit-learn pipeline.

```python
pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", dt)
    ]
)
```

This ensures that the same preprocessing operations are applied consistently during training and prediction.

## 📊 Model Evaluation

The initial model is evaluated using:

- **F1 Score:** Balances precision and recall.
- **Precision:** Measures how many predicted purchase sessions actually resulted in purchases.
- **Recall:** Measures how many actual purchase sessions were identified.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.
- **Classification Report:** Provides precision, recall, F1 score, and support for each class.

### Initial Model Results

| Metric | Score |
|---|---:|
| Accuracy | 84.71% |
| Precision — Purchase class | 50.40% |
| Recall — Purchase class | 83.25% |
| F1 Score — Purchase class | 62.78% |

The model achieves a purchase-class recall of approximately **83.25%**, meaning it identifies a substantial proportion of actual purchase sessions. Its purchase-class precision is lower, indicating that some sessions predicted as purchases do not result in purchases.

### Confusion Matrix

```text
[[1771, 313],
 [  64, 318]]
```

| | Predicted No Purchase | Predicted Purchase |
|---|---:|---:|
| Actual No Purchase | 1,771 | 313 |
| Actual Purchase | 64 | 318 |

## 🔍 Hyperparameter Tuning

The notebook uses `GridSearchCV` to identify better decision tree configurations.

### Parameters Tested

```python
param_grid = {
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [20, 30, 50]
}
```

The search uses:

- Five-fold cross-validation
- F1 score as the optimization metric
- Parallel processing with `n_jobs=-1`

### Best Configuration

The best configuration from the cross-validation search was:

| Parameter | Best Value |
|---|---:|
| `model__max_depth` | 4 |
| `model__min_samples_leaf` | 50 |
| Best cross-validation F1 score | 63.44% |

The tuned pipeline achieved an F1 score of approximately **62.29%** and an accuracy of **84.39%** on the held-out test set.

These results show that optimizing cross-validation F1 does not necessarily improve every test metric. Model selection should prioritize the business objective, such as identifying likely purchases while controlling false positives.

## 📁 Project Structure

```text
ShopSmart/
├── shop_smart.ipynb
├── shop_smart_ecommerce.csv
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ShopSmart
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open `shop_smart.ipynb` and execute the cells sequentially.

**Important:** The notebook currently loads the dataset using `pd.read_csv("online_shoppers.csv")`. Since the supplied dataset is named `shop_smart_ecommerce.csv`, update the loading statement to:

```python
df = pd.read_csv("shop_smart_ecommerce.csv")
```

Ensure the CSV file is located in the notebook's working directory.

## 📚 Learning Outcomes

- Binary classification
- Data preprocessing with Scikit-learn
- Numerical feature scaling
- Categorical feature encoding
- Handling class imbalance
- Decision Tree Classification
- Scikit-learn pipelines
- F1 score and confusion matrix interpretation
- Hyperparameter tuning with GridSearchCV
- Cross-validation

## ⚠️ Limitations and Future Improvements

- Compare the decision tree with Logistic Regression, Random Forest, and Gradient Boosting.
- Evaluate precision-recall trade-offs for purchase prediction.
- Use cross-validation for reliable model selection.
- Investigate feature importance to understand the strongest purchase signals.
- Consider probability calibration if the model will support business decisions.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (AI & ML)

---

⭐ If you find this project useful, consider starring the repository!
