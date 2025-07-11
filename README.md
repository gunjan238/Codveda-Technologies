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
## 🧹 Level 2 Data Analysis Tasks

## ✅ Task 1 – Regression Analysis

**Description:**  
Performed simple linear regression to predict one variable based on another.

**Objectives:**

- Split the dataset into training and testing sets
- Fit a linear regression model using scikit-learn
- Interpret model coefficients
- Evaluate model performance using metrics such as R-squared and Mean Squared Error (MSE)


**Dataset:**  
Boston Housing dataset, including attributes of houses in Boston suburbs and their median prices.

---

## ✅ Task 2 – Time Series Analysis

**Description:**  
Analyzed time-series data (e.g. stock prices) to detect trends and seasonality.

**Objectives:**

- Import and explore financial time series data using Yahoo Finance
- Visualize time series data to identify trends and patterns
- Decompose the series into trend, seasonality, and residuals using statsmodels
- Apply moving average smoothing to reveal long-term trends


**Dataset:**  
Daily closing stock prices for companies like Amazon (AMZN) sourced from Yahoo Finance.

---

## ✅ Task 3 – Clustering Analysis (K-Means)

**Description:**  
Implemented K-Means clustering to group similar data points based on feature similarities.

**Objectives:**

- Standardize the dataset using StandardScaler
- Apply K-Means clustering
- Determine the optimal number of clusters using the elbow method
- Visualize clusters in 2D scatter plots
- Analyze cluster centers to understand customer segments

**Dataset:**  
Mall Customers dataset, containing features like customer age, annual income, and spending score.

---
## 🧹 Level 3 Data Analysis Tasks

## ✅ Task 1 – Predictive Modeling (Classification)

 **Description**
 Performed basic text analysis and sentiment analysis on textual data.
 
 **Objectives**
- Preprocss the data
- Train and test multiple classification
- Evaluate models using accuracy, precision, recall and F1-score
- Perform hyperparameter tuning using grid search.


**Dataset:**  
UCI Credit Card data

---

## ✅ Task 2 – Building Dashboard using PowerBI

**Description:**  
Create dashboard to provide insights from the data.

**Objectives:**

- Import and clean the dataset in Power BI.
- Create interactive visualizations (e.g., bar charts, line
graphs, maps).
- Set up filters and slicers for interactive exploration.
- Publish the dashboard and share it with others.

*<img width="768" height="431" alt="Airbnb Dashboard" src="https://github.com/user-attachments/assets/8fbd1394-f3d9-4e55-b1e7-938d0850729e" />

**Dataset:** Airbnb NYC data

---

## ✅ Task 3 – Natural Language Processing(NLP) - Sentiment Analysis

**Description:**  
Perform sentiment analysis on textual data.

**Objectives:**

- Preprocess text data (tokenization, removing stopwords, and stemming/lemmatization).
- Use nltk or TextBlob for sentiment analysis.
- Visualize the sentiment distribution and word frequencies using word clouds.

**Dataset:**  
Tweets data

---



---
## 🛠 Tools & Libraries

- Python 3.8+
- pandas
- matplotlib
- seaborn
- scikit-learn
- numpy
- statsmodels
- yfinance
- Jupyter Notebook or any Python IDE
- nltk
- textblob

All the Datasets uploaded 

**Attached all scripts of `Level 2` in `Level 2` folder**

---



