# Sentiment Analysis using Spark and Kafka

## Course

Hardware and Software for Big Data

## Author

**Deepak Kushwaha**

------------------------------------------------------------------------

# Project Overview

This project implements a **sentiment analysis system** for large-scale
Twitter data using **Apache Spark** and **Apache Kafka**. The goal of
the project is to perform **multiclass classification** on tweets to
determine their sentiment.

------------------------------------------------------------------------

# Dataset

Sentiment Dataset with 1 Million Tweets\
https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets

------------------------------------------------------------------------

# Technologies Used

-   Apache Spark
-   Apache Kafka
-   Python (PySpark)
-   Spark NLP

------------------------------------------------------------------------

# Project Architecture

1.  Kafka Producer streams tweets.
2.  Kafka Broker handles streaming messages.
3.  Spark Streaming consumes data from Kafka.
4.  Spark MLlib trains and evaluates the sentiment classification model.

------------------------------------------------------------------------

# Methodology

1.  Data ingestion using Kafka.
2.  Data preprocessing (cleaning, tokenization, stopword removal).
3.  Feature extraction.
4.  Sentiment classification using Spark ML models.

------------------------------------------------------------------------

# Evaluation Metrics

-   Accuracy
-   Precision
-   Recall

------------------------------------------------------------------------

# Project Structure

project/ │ ├── data/ ├── kafka/ ├── spark/ ├── models/ └── README.md

------------------------------------------------------------------------

# Conclusion

This project demonstrates how Spark and Kafka can be used together to
build scalable pipelines for sentiment analysis on large datasets.

------------------------------------------------------------------------

# Acknowledgment

Project developed for the **Hardware and Software for Big Data** course.
