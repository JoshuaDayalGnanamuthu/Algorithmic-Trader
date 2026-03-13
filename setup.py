import mysql.connector as my
import os
from dotenv import load_dotenv

load_dotenv("credentials.env")
USERNAME = os.getenv("DATABASE_USERNAME")
PASSWORD = os.getenv("DATABASE_PASSWORD")
HOST = os.getenv("DATABASE_HOST")

conn = my.connect(host=HOST, user=USERNAME, password=PASSWORD)

if conn.is_connected():
    print("Connected to MySQL database")
else:
    print("Failed to connect to MySQL database")

cursor = conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS ALGOTRADER")
cursor.execute("USE ALGOTRADER")
cursor.execute("""CREATE TABLE IF NOT EXISTS TRADES (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    TICKER VARCHAR(10) NOT NULL,
    SIDE ENUM('buy', 'sell') NOT NULL,
    QUANTITY FLOAT NOT NULL,
    PRICE FLOAT NOT NULL,
    TOTAL FLOAT NOT NULL
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS ERRORS (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    TYPE VARCHAR(10) DEFAULT 'ERROR',
    MESSAGE TEXT NOT NULL
)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS EVENTS (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    TYPE VARCHAR(10) DEFAULT 'EVENT',
    MESSAGE TEXT NOT NULL
)""")
print("Database and table setup completed.")

cursor.close()
conn.close()

print("MySQL connection closed.")