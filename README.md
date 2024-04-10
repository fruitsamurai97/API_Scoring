# Fast API for Client Classification

This repository contains the Fast API for predicting the class membership probability of a client based on a pre-trained LightGBM model.

## Structure

- `app`: This directory contains the Flask app.
  - `__init__.py`: Initialization file for the Flask application.
  - `routes.py`: Contains all the routes/endpoints for the API.
- `tests`: Contains tests for the application routes.
  - `test_routes`: Test cases for routes functionality.
- `Azure_container_key.txt`: Contains the Azure container access key.
- `requirements.txt`: Lists all Python library dependencies.
- `run.py`: The entry point to run the Flask application.
- `startup.sh`: Shell script to set up and start the API server.

## Endpoints

### `/client`
- Method: GET
- Input: None
- Output: Returns a list of unique client IDs from the test dataframe.
- Function: Provides a list of client IDs that can be selected for predictions.

### `/predict`
- Method: GET
- Input: ID (Client ID)
- Output: Probability of payback credit for the specified client.
- Function: Performs a prediction using the LightGBM model for the given client ID.

### `/explain`
- Method: GET
- Input: ID (Client ID)
- Output: Explanation of the prediction using LIME.
- Function: Provides an explanation for the model's prediction for a specific client ID.

### `/info`
- Method: GET
- Input: ID (Client ID)
- Output: Important attributes for the client.
- Function: Retrieves important information attributes about the client.

### `/feature`
- Method: GET
- Input: Feature Name (column name from dataframe)
- Output: Values for the specified feature across the entire dataset.
- Function: Gets the values of a specific feature for all clients in the dataset.

## Routes.py Functions

Below is a brief overview of the functions defined in `routes.py`:

- `get_client()`: Serves the `/client` endpoint to provide a list of client IDs.
- `predict()`: Serves the `/predict` endpoint to calculate and return the prediction probabilities.
- `explain()`: Serves the `/explain` endpoint to provide explanations for predictions.
- `get_info()`: Serves the `/info` endpoint to provide detailed client information.
- `get_feature()`: Serves the `/feature` endpoint to provide feature values across the dataset.




