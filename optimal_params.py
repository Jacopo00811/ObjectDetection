import pandas as pd
import sqlite3

# Load CSV data into a DataFrame
df = pd.read_csv('parameter_results.csv')

# Connect to SQLite (or create a database file)
conn = sqlite3.connect('params.db')
df.to_sql('params', conn, if_exists='replace', index=False)

# Run the query
query = """
    SELECT Image, Alpha, Beta, [Min Score], Recall, MABO
    FROM params
    WHERE (Recall + MABO) / 2 = (
        SELECT MAX((Recall + MABO) / 2)
        FROM params AS p2
        WHERE p2.Image = params.Image
    );
"""
optimal_params = pd.read_sql_query(query, conn)
conn.close()

print(optimal_params)
