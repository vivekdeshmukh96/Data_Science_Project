from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('uploadpage.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f'Request Method: {request.method}')  # Debug line
    print(f'Files: {request.files}')  # Debug line

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Load dataset
            data = pd.read_csv(file_path)

            # Check for necessary columns
            required_columns = [
                'Team_Size', 'Promotions', 'Remote_Work_Frequency', 'Resigned', 
                'Department', 'Job_Title', 'Education_Level', 
                'Monthly_Salary', 'Age', 'Work_Hours_Per_Week', 
                'Projects_Handled', 'Overtime_Hours', 'Sick_Days', 
                'Training_Hours', 'Employee_Satisfaction_Score'
            ]

            # Validate that required columns exist in the uploaded data
            for col in required_columns:
                if col not in data.columns:
                    flash(f'Missing column: {col}')
                    return redirect(request.url)

            # Impute missing values (numerical and categorical)
            num_cols = ['Team_Size', 'Promotions']  # Example numerical columns
            num_imputer = SimpleImputer(strategy='mean')
            data[num_cols] = num_imputer.fit_transform(data[num_cols])

            cat_cols = ['Remote_Work_Frequency', 'Resigned']  # Example categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

            # One-hot encoding for categorical columns
            data = pd.get_dummies(data, columns=['Department', 'Job_Title', 'Education_Level'], drop_first=True)

            # Label Encoding for binary variables
            if 'Gender' in data.columns:
                label_encoder = LabelEncoder()
                data['Gender'] = label_encoder.fit_transform(data['Gender'])

            # Scaling numerical columns
            num_cols = ['Monthly_Salary', 'Age', 'Work_Hours_Per_Week', 'Projects_Handled',
                        'Overtime_Hours', 'Sick_Days', 'Training_Hours', 'Employee_Satisfaction_Score']
            standard_scaler = StandardScaler()
            data[num_cols] = standard_scaler.fit_transform(data[num_cols])

            # Perform Linear Regression
            X = data.drop('Employee_Satisfaction_Score', axis=1)  # Assuming 'Employee_Satisfaction_Score' is the target
            y = data['Employee_Satisfaction_Score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store the results in session or pass them directly to the results page
            return redirect(url_for('results', mse=round(mse, 2), r2=round(r2, 2)))

        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)

@app.route('/results')
def results():
    mse = request.args.get('mse')
    r2 = request.args.get('r2')
    return render_template('processed.html', mse=mse, r2=r2)
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensuring the app runs on port 5000
