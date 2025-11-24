# JP Morgan Chase – Quantitative Research Virtual Job Simulation

A comprehensive project completed as part of the JPMorgan Chase & Co. Quantitative Research Virtual Job Simulation on Forage.
This repository contains forecasting models, valuation prototypes, utilities, and documentation reflecting a real-world Quant Research workflow.

---

## Overview

This project focuses on:

- Time-series forecasting of natural gas prices

- Gas storage contract pricing using forecasted curves

- Scenario & sensitivity analysis

- Building modular, production-style Python scripts

The simulation replicates the type of analytical and modeling tasks handled by JPMC’s Quantitative Research team.

---

## Natural Gas Price Forecasting

This script performs monthly natural gas price forecasting using the Holt-Winters (Additive Trend + Additive Seasonality) model.

### Key Features

- Loads and preprocesses monthly pricing data

- Fits Holt-Winters model

- Generates 12-month forward price forecast

- Includes: estimate_price(date)

A utility to fetch forecasted prices for any given date.

### Outputs

- Forecast curves

- Trend & seasonality decomposition

- Plots saved in /results/

---

## Gas Storage Contract Pricing Model

Simulates gas storage operations and calculates strategy profitability.

### Model Capabilities

- Injection, withdrawal, and hold actions

- Capacity limits & rate constraints

- Daily operational/storage costs

- NPV-based valuation using the forecast price curve

- Validations for dates & storage logic

### Purpose

This model demonstrates valuation thinking in energy markets, a key area for real-world quant teams.

---

## Loan Default & Expected Loss Model

Estimates borrower Probability of Default (PD) and computes Expected Loss (EL) for loan exposures.

### Features

- Loads borrower financial data from CSV

- Trains Logistic Regression and Random Forest models

- Scales numerical inputs for improved model performance

- Evaluates models using AUC, Brier Score, and PR-AUC

- Predicts borrower-level PD values

- Calculates Expected Loss using: EL = PD × EAD × (1 − Recovery Rate)

- Includes a reusable expected_loss() function for any loan record

### Purpose

Provides a simple, interpretable credit-risk model for estimating borrower default probability and expected loss, supporting risk assessment and portfolio provisioning workflows.

---

## MSE-Based Quantization & Credit Rating Model

Creates optimized rating buckets from continuous credit scores using Mean Squared Error (MSE) quantization, enabling category-based risk modeling.

### Features

- Loads borrower credit-score and default data from CSV

- Aggregates records by unique score values for efficient processing

- Uses dynamic programming to minimize within-bucket Mean Squared Error

- Generates optimal bucket boundaries for any chosen number of ratings

- Assigns ratings where lower rating = better credit quality

- Computes bucket-level statistics:
  - Record count
  - Defaults
  - Probability of Default (PD)

- Saves final rating map to CSV for reuse with future datasets

- Includes a reusable map_score_to_rating() function for new incoming data

### Purpose

Provides a general, data-driven quantization framework for transforming continuous credit scores into categorical risk ratings.
Supports probability-of-default modeling, segmentation, and credit-risk decision workflows where model architectures require discrete input features.

---

## Tech Stack & Dependencies

### Languages & Libraries

- Python

- Pandas

- NumPy

- Statsmodels

- Matplotlib

### Install Dependencies

pip install pandas numpy matplotlib statsmodels

---

## Installation & Setup

### Clone the repository:

```bash
git clone https://github.com/ArinAyare/Quantitive-Research-JP-Morgan-Chase
cd Quantitive-Research-JP-Morgan-Chase
```
