# Customer Churn Prediction App

## Overview

This Streamlit application predicts customer churn for a bank using various machine learning models. It provides an interactive interface for selecting customers, viewing their details, and getting predictions on their likelihood to churn. The app also generates explanations for the predictions and personalized emails for customer retention.

## Features

- Interactive customer selection
- Display of customer details
- Churn prediction using multiple machine learning models
- Visualization of churn probability
- Comparison of predictions across different models
- Percentile analysis of customer attributes
- AI-generated explanation of churn prediction
- AI-generated personalized retention email

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your GROQ API key:
   - Create a `.streamlit/secrets.toml` file in the project directory
   - Add your GROQ API key to this file:
     ```
     GROQ_API_KEY = "your-api-key-here"
     ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit

3. Use the dropdown menu to select a customer.

4. View the customer's details, churn prediction, and other insights.

5. Scroll down to see the AI-generated explanation and personalized email.

## Files

- `streamlit_app.py`: Main Streamlit application script
- `utils.py`: Utility functions for creating charts and visualizations
- `churn.csv`: Dataset containing customer information
- `xgb_model.pkl`, `nb_model.pkl`, etc.: Trained machine learning models

## Models Used

- XGBoost
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly
- openai

## Note

This application uses the GROQ API for generating explanations and emails. Ensure you have a valid API key and sufficient credits to use these features.

## Contributing

Contributions to improve the app are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

[MIT License](LICENSE)
