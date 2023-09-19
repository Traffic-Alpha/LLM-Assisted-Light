'''
@Author: WANG Maonan
@Date: 2023-09-19 16:48:27
@Description: 读取 SQLite 的数据, 并计算相似度, 数字越小越相似
@LastEditTime: 2023-09-19 19:59:18
'''
import sqlite3
from tshub.utils.get_abs_path import get_abs_path
from utils.junction_similarity import calculate_similarity

path_convert = get_abs_path(__file__)

if __name__ == '__main__':
    database_path = path_convert("./junction.db")
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    rows = cursor.execute("SELECT * FROM junctionINFO;").fetchall()
    scenario_anchor = rows[20]
    anchor_similarity = calculate_similarity(scenario_anchor)
    for row_index, _scenario in enumerate(rows):
        similarity_score = anchor_similarity(_scenario)
        print(f'Index, {row_index}; Similarity Score, {similarity_score}')