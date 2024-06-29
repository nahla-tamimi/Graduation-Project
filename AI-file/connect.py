import mysql.connector
import pandas as pd

connection = mysql.connector.connect(host="localhost",
    user="optifisc_hireadmin",
    passwd="A$C.cC9W}}i9",
    database="optifisc_hire"
)
if connection.is_connected():
    print("MySQL Connected...")
else:
    print("Connection Failed.")

cursor = connection.cursor()


# Create a cursor object
cursor = connection.cursor()

# Execute SQL query to fetch data from predict_score table
cursor.execute("SELECT * FROM predict_score")

# Fetch all rows from the result set
rows = cursor.fetchall()

# Get column names from the cursor description
columns = [col[0] for col in cursor.description]

# Create a DataFrame from the fetched rows with column names
df = pd.DataFrame(rows, columns=columns)

# Specify the path where you want to save the Excel file
excel_file_path = r"Frame_company_data.xlsx"

# Write DataFrame to Excel file
df.to_excel(excel_file_path, index=False)

# Close cursor and connection
cursor.close()
connection.close()

print("Data saved to Excel successfully.")



# cursor.execute("SHOW TABLES")

# # Fetch all table names
# tables = cursor.fetchall()

# # Print table names
# for table in tables:
#     print(table[0])

# # Close cursor and connection
# cursor.close()
# connection.close()

