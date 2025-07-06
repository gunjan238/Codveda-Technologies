# Codveda-Technologies

## 🧹 Level 1 Data Analysis Task – Airbnb NYC Listings

This repository contains data analysis tasks using Python, pandas, matplotlib, and seaborn. The dataset used is the **Airbnb NYC 2019 listings**, which contains over 48,000 rows of short-term rental information in New York City.

---

## 📁 Dataset

- **Source**: [Airbnb Open Data]([http://insideairbnb.com/get-the-data.html](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data))
- **File**: `AB_NYC_2019.csv`
- **Features**: `name`, `host_name`, `neighbourhood`, `room_type`, `price`, `number_of_reviews`, `reviews_per_month`, `last_review`, etc.

---

## ✅ Level 1 Tasks

### 🔧 Task 1: Data Cleaning & Preprocessing

> **Objective**: Prepare the dataset for analysis by cleaning and standardizing.

- ✅ Load dataset using pandas
- ✅ Identify and handle missing values (e.g. fill or drop)
- ✅ Remove duplicate rows
- ✅ Standardize formats (e.g., date format, categorical labels)
- ✅ Convert `last_review` to datetime
- ✅ Fill missing values in `reviews_per_month`, `name`, `host_name`

📂 Script: `Task1.ipynb`

---

### 📊 Task 2: Exploratory Data Analysis (EDA)

> **Objective**: Explore the dataset to find patterns, trends, and summary statistics.

- ✅ Calculate summary statistics (mean, median, mode, std)
- ✅ Check distributions of numerical features
- ✅ Identify relationships using scatter plots and correlation heatmaps
- ✅ Analyze room types, prices, neighbourhoods, reviews

📂 Script: `Task2.ipynb`

🔍 Visuals:
- Histograms
- Boxplots
- Correlation matrix

---

### 📈 Task 3: Basic Data Visualization

> **Objective**: Visualize insights with charts and export them for reports.

- ✅ Bar plot: Number of listings by room type (with legend)
- ✅ Line chart: Average price by neighbourhood group
- ✅ Scatter plot: Price vs. number of reviews
- ✅ Customize plot titles, axes, and legends
- ✅ Export plots to PNG files

📂 Script: `Task3.ipynb`  
📁 Outputs: `bar_plot.png`, `line_chart.png`, `scatter_plot.png`

---

## 🛠 Tools & Libraries

- Python 3.8+
- pandas
- matplotlib
- seaborn
- Jupyter Notebook or any Python IDE

---



