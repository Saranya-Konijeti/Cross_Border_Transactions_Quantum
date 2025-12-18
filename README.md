# Fraud Detection Analytics: Classical vs Quantum SVM

## Overview
This project is an advanced fraud detection analytics dashboard built using Streamlit. It provides an interactive environment to analyze financial transaction data and compare the performance of Classical Support Vector Machines (SVM) with an experimental Quantum SVM approach using PennyLane.

The application enables real-time data filtering, animated processing, detailed visual analytics, and side-by-side algorithm comparison to study how quantum machine learning techniques can enhance fraud detection.

## Key Features
- Interactive Streamlit dashboard with real-time filters
- Classical SVM implementation with robust preprocessing
- Experimental Quantum SVM using quantum feature encoding and quantum kernels
- PCA-based dimensionality reduction and feature scaling
- Comprehensive performance metrics including Accuracy, Precision, Recall, F1-score, and AUC
- Visualizations using Plotly (ROC curves, confusion matrices, distributions)
- Animated processing flow using Lottie animations
- Integrated AI assistant restricted to project-specific explanations

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- PennyLane (Quantum Machine Learning)
- NumPy and Pandas
- Plotly
- Google Gemini API (for contextual AI assistant)

## Dataset Requirements
The uploaded CSV file must contain the following columns:
- TransactionID
- Amount
- CountryRisk
- TimeOfDay (Day/Night)
- SenderBlacklisted (0 or 1)
- SenderAgeDays
- Label (0 = Genuine, 1 = Fraud)

## Application Modes
- Classical SVM: Traditional machine learning-based fraud detection
- Quantum SVM (Experimental): Quantum-enhanced fraud detection using simulated quantum circuits
- Compare Both Algorithms: Side-by-side comparison of classical and quantum models

## Purpose
This project is intended for academic research and learning, demonstrating how quantum machine learning concepts can be applied to real-world financial fraud detection problems and compared against classical approaches.

## Note
Quantum components are simulated and intended for experimental and educational use rather than production deployment.

## Author
Konijeti Saranya
