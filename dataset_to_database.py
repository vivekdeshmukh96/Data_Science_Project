import pandas as pd
import pymysql

# Load CSV file
df = pd.read_csv("cleaned_cutoff_data.csv")

# Database connection details
HOST = "localhost"
USER = "root"
PASSWORD = "root"  # Replace with your actual MySQL password
DATABASE = "college_data"

# Connect to MySQL
def connect_db():
    return pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)

# Insert data into MySQL
def insert_data(df):
    connection = connect_db()
    cursor = connection.cursor()
    
    insert_query = """
    INSERT INTO college_cutoffs (College_Code, College_Name, Branch_Code, Branch_Name, Status, Level, Stage, Category, Merit_Number, Cutoff) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    data_tuples = [tuple(row) for row in df.to_numpy()]
    
    try:
        cursor.executemany(insert_query, data_tuples)
        connection.commit()
        print("Data inserted successfully!")
    except pymysql.Error as e:
        print("Error inserting data:", e)
    finally:
        cursor.close()
        connection.close()

# Run the insertion
insert_data(df)
print(df['Status'].apply(len).max())  # Finds the longest string length
print(df['Status'].unique())  # Lists unique values
print(df.isna().sum())  # Shows number of NaNs per column
df = df.where(pd.notna(df), None)

