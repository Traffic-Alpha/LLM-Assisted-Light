'''
@Author: WANG Maonan
@Date: 2023-09-19 14:58:09
@Description: 尝试使用 SQLite
@LastEditTime: 2023-09-19 15:16:03
'''
import sqlite3

# Step 1, create a connection to SQLite Database
connection = sqlite3.connect("aquarium.db")

# Step 2 — Adding Data to the SQLite Database
cursor = connection.cursor()
cursor.execute("CREATE TABLE fish (name TEXT, species TEXT, tank_number INTEGER)")

cursor.execute("INSERT INTO fish VALUES ('Sammy', 'shark', 1)")
cursor.execute("INSERT INTO fish VALUES ('Jamie', 'cuttlefish', 7)")
connection.commit()

# Step 3 — Reading Data from the SQLite Database
rows = cursor.execute("SELECT name, species, tank_number FROM fish").fetchall()
print(rows)

target_fish_name = "Jamie"
rows = cursor.execute(
    "SELECT name, species, tank_number FROM fish WHERE name = ?",
    (target_fish_name,),
).fetchall()
print(rows)

# Step 4 — Modifying Data in the SQLite Database
new_tank_number = 2 # update tank_number
moved_fish_name = "Sammy"
cursor.execute(
    "UPDATE fish SET tank_number = ? WHERE name = ?",
    (new_tank_number, moved_fish_name)
)
connection.commit()

rows = cursor.execute("SELECT name, species, tank_number FROM fish").fetchall()
print(rows)

connection.close()