Product Categorization Model
Overview
This repository contains a Jupyter notebook designed for classifying products into various categories based on their descriptions. The project involves data preprocessing, model training, and evaluation using several machine learning algorithms and deep learning techniques. It aims to provide accurate product categorization to enhance eCommerce experiences.

Dataset
The dataset used in this project consists of product information from an eCommerce platform. The main file, train_product_data.csv, includes the following columns:

uniq_id: Unique identifier for each product.
crawl_timestamp: Timestamp of when the product data was scraped.
product_url: Direct link to the product page.
product_name: Title of the product.
product_category_tree: Hierarchical category of the product.
pid: Platform-specific product identifier.
retail_price: Original price before discounts.
discounted_price: Price after discounts.
description: Detailed product description.
product_rating: Rating given by customers.
overall_rating: Aggregate rating across various platforms.
brand: Brand name of the product.
product_specifications: Detailed product specifications.
Additional files include test_data.csv for model testing and test_results.xlsx containing the true labels for evaluation.

Data Preprocessing
Data preprocessing steps include:

Handling Missing Values:

Filled missing values in the brand column with the most common brand.
Dropped rows with missing description, image, and product_specifications.
Filled missing values in retail_price and discounted_price with median values.
Text Cleaning:

Cleaned the description field to remove special characters, extra whitespace, and convert text to lowercase.
Feature Engineering:

Transformed product descriptions into TF-IDF features to be used for model training.
Label Encoding:

Encoded the product_category_tree labels into numeric values using LabelEncoder.
Models
1. Logistic Regression
Description: A linear classification model used as a baseline for comparison.
Implementation: Utilized logistic regression with TF-IDF features to train the model and predict product categories.
2. Random Forest
Description: An ensemble method combining multiple decision trees to improve prediction accuracy.
Implementation: Trained a Random Forest classifier with 100 trees and evaluated its performance on the validation set.
3. Deep Learning
Description: A neural network model using Dense and Dropout layers to capture complex patterns in product descriptions.
Implementation: Built and trained a Sequential model with Dense layers and Dropout for regularization.
Model Evaluation
Models were evaluated using accuracy, precision, recall, and F1 score. The Deep Learning model achieved high accuracy (0.98) on the validation set. However, the F1 score dropped to 0 on the test data, indicating potential issues:
