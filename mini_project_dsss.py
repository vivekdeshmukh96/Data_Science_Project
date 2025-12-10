from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

app = Flask(__name__)

# Route for rendering the HTML page (frontend)
@app.route('/')
def index():
    return render_template('mini_pr.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        if file and file.filename.endswith('.csv'):
            # Load the CSV file
            data = pd.read_csv(file)
            
            # Apply the preprocessing as per your script
            # (Handling missing values, outliers, etc.)
            data['Add-on Total'].fillna(data['Add-on Total'].median(), inplace=True)
            data['Add-ons Purchased'].fillna(data['Add-ons Purchased'].mode()[0], inplace=True)
            data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
            
            # Detect and fix outliers (as per your function)
            def fix_outliers_iqr(df, column):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                return df

            columns_to_fix = ['Total Price', 'Add-on Total']
            for column in columns_to_fix:
                data = fix_outliers_iqr(data, column)

            # Define features and target
            categorical_features = ['Gender', 'Loyalty Member', 'Order Status', 'Payment Method', 'Shipping Type']
            numerical_features = ['Age', 'Rating', 'Unit Price', 'Quantity', 'Add-on Total']

            X = data.drop(['Total Price', 'Customer ID'], axis=1)
            y = data['Total Price']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
                ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            # Model evaluation
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Send the evaluation results back to the frontend
            results = {
                "r2": r2,
                "mae": mae,
                "mse": mse
            }

            return jsonify(results)

        return jsonify({"error": "Unsupported file format"})
    return render_template('uploadpage.html')

@app.route('/contact')
def contact():
    return render_template('contact_us_page.html')

@app.route('/about')
def about():
    return render_template('About_us_page.html')

if __name__ == '__main__':
    app.run(debug=True)
