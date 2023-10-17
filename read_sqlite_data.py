'''
@Author: WANG Maonan
@Date: 2023-09-19 16:48:27
@Description: 读取 SQLite 的数据, 并计算相似度, 数字越小越相似
@LastEditTime: 2023-09-19 20:54:26
'''
import sqlite3
from tshub.utils.get_abs_path import get_abs_path
from utils.junction_similarity import calculate_similarity, find_min_indices
from utils.convert_sql2description import convert_description

path_convert = get_abs_path(__file__)

if __name__ == '__main__':
    database_path = path_convert("./junction.db")
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    rows = cursor.execute("SELECT * FROM junctionINFO;").fetchall()
    scenario_anchor = rows[20]
    similarity_score_list = list()
    anchor_similarity = calculate_similarity(scenario_anchor)
    for row_index, _scenario in enumerate(rows):
        similarity_score = anchor_similarity(_scenario)
        print(f'Index, {row_index}; Similarity Score, {similarity_score}')
        similarity_score_list.append(similarity_score)
    
    # 找出最接近的前 n 个例子
    similarity_indexs = find_min_indices(similarity_score_list, 2)
    for _i in similarity_indexs:
        print(convert_description(rows[_i])) # 将 sql 内容转换为场景的描述