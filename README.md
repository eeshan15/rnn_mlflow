# Basic Sine Wave Predictor (RNN + MLflow + DVC)

This project demonstrates a basic implementation of MLOps tools (DVC and MLflow) integrated with a PyTorch RNN model. The goal is to predict the next value in a Sine wave sequence.

## Project Overview

* **Model:** Recurrent Neural Network (RNN) built with PyTorch.
* **Data Versioning:** DVC (Data Version Control) is used to track the dataset.
* **Experiment Tracking:** MLflow is used to log parameters (learning rate, hidden size) and metrics (loss).

## Prerequisites

Ensure you have Python installed. You will need the following libraries:

* torch
* numpy
* pandas
* matplotlib
* mlflow
* dvc

## Setup and Installation

1.  **Clone the repository (if applicable) or create a new directory:**
    ```bash
    mkdir rnn_mlflow_dvc_basic
    cd rnn_mlflow_dvc_basic
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy pandas matplotlib mlflow dvc
    ```

3.  **Initialize Git and DVC:**
    ```bash
    git init
    dvc init
    ```

## Data Management (DVC)

We generate synthetic data for this project and track it using DVC instead of Git.

1.  **Generate the dataset:**
    Run the generation script to create `data/sine_wave.csv`.
    ```bash
    python generate_data.py
    ```

2.  **Track data with DVC:**
    This command adds the CSV file to DVC and creates a `.dvc` placeholder file.
    ```bash
    dvc add data/sine_wave.csv
    ```

3.  **Track DVC files with Git:**
    We track the pointer file (`.dvc`) and the `.gitignore` updated by DVC.
    ```bash
    git add data/sine_wave.csv.dvc data/.gitignore
    git commit -m "Initialize data tracking with DVC"
    ```

## Training and Experiment Tracking (MLflow)

The training script trains the RNN model and logs the run details to MLflow.

1.  **Run the training script:**
    ```bash
    python train.py
    ```
    This will:
    * Load data from `data/sine_wave.csv`.
    * Train the RNN model for the specified epochs.
    * Log parameters (Hidden Size, Learning Rate) and Metric (Loss) to MLflow.
    * Save the final model as an MLflow artifact.

2.  **View Results in MLflow UI:**
    To visualize the training loss and compare experiments, run the MLflow server:
    ```bash
    mlflow ui
    ```
    Open your browser and navigate to `http://127.0.0.1:5000`.

## Project Structure

* `data/`: Contains the dataset (managed by DVC).
* `generate_data.py`: Script to generate synthetic Sine wave data.
* `train.py`: Main script for training the RNN and logging to MLflow.
* `mlruns/`: Directory where MLflow stores experiment logs locally.
* `data/sine_wave.csv.dvc`: The DVC pointer file for the dataset.

## Workflow Summary

1.  Modify `generate_data.py` to change data parameters.
2.  Run `python generate_data.py` to update the CSV.
3.  Run `dvc add data/sine_wave.csv` to update the data version.
4.  Run `python train.py` to train the model on new data.
5.  Compare results in MLflow.
