import os
import pandas as pd
import warnings

def get_sql_type(excel_type):
    """将Excel类型转换为SQL类型"""
    type_map = {
        'STRING': 'VARCHAR(255)',
        'INT': 'INT',
        'FLOAT': 'FLOAT'
    }
    return type_map.get(excel_type, f'{excel_type} -- 类型需要手动修正')

def generate_create_table_sql(table_name, columns):
    """生成创建表的SQL语句"""
    sql = f"CREATE TABLE {table_name} (\n"
    sql += ",\n".join([f"    {col_name} {get_sql_type(col_type)}" for col_name, col_type in columns])
    sql += "\n);\n"
    return sql

def generate_insert_sql(table_name, columns, data):
    """生成插入数据的SQL语句"""
    column_names = [col_name for col_name, _ in columns]
    sql = ""
    for _, row in data.iterrows():
        values_str = ", ".join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
        sql += f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({values_str});\n"
    return sql

def process_excel_file(file_path):
    """处理单个Excel文件，生成对应的SQL文件"""
    try:
        # 读取特定工作表
        df = pd.read_excel(file_path, sheet_name="工作表1")

        # 提取表格信息
        table_name = os.path.splitext(os.path.basename(file_path))[0]
        column_names = df.iloc[0]  # 第2行是列名
        column_types = df.iloc[1]  # 第3行是列类型
        columns = [(column_names.iloc[i], column_types.iloc[i]) for i in range(len(column_names))]
        data = df.iloc[3:].reset_index(drop=True)
        data.columns = column_names

        # 生成SQL语句
        create_table_sql = generate_create_table_sql(table_name, columns)
        insert_sql = generate_insert_sql(table_name, columns, data)

        # 写入SQL文件
        sql_file_path = f"{table_name}.sql"
        with open(sql_file_path, 'w', encoding='utf-8') as sql_file:
            sql_file.write(create_table_sql)
            sql_file.write(insert_sql)
            print(f"Successfully generated SQL file: {file_path} -> {sql_file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    """主函数，处理当前目录下的所有Excel文件"""
    # 过滤警告信息
    warnings.simplefilter("ignore", UserWarning)
    
    for file_name in os.listdir('.'):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            process_excel_file(file_name)

if __name__ == "__main__":
    main()
