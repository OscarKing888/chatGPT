import tkinter as tk
from tkinter import ttk
import os

# 读取文件列表
def read_file_list(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 获取根目录
    root_dir = lines[0].strip()

    # 获取文件名和翻译名
    file_pairs = []
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                original_name, translated_name = parts
                file_pairs.append((original_name, translated_name))
    return root_dir, file_pairs

# 打开文件
def open_file(root_dir, filename):
    full_path = os.path.join(root_dir, filename)
    full_path = os.path.normpath(full_path)
    if os.path.exists(full_path):
        os.startfile(full_path)
    else:
        print(f"文件未找到: {full_path}")

# 创建 GUI
def create_gui(root_dir, file_pairs):
    root = tk.Tk()
    root.title("文件列表")

    # 创建顶层框架
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X)

    # 添加过滤器
    filter_label = tk.Label(top_frame, text="过滤器：")
    filter_label.pack(side=tk.LEFT, padx=5, pady=5)

    filter_entry = tk.Entry(top_frame)
    filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

    # 创建表格的容器
    tree_frame = tk.Frame(root)
    tree_frame.pack(fill=tk.BOTH, expand=True)

    # 添加滚动条
    tree_scrollbar_y = ttk.Scrollbar(tree_frame, orient='vertical')
    tree_scrollbar_x = ttk.Scrollbar(tree_frame, orient='horizontal')

    # 创建表格
    columns = ('Original', 'Translated', 'Open')
    tree = ttk.Treeview(tree_frame, columns=columns, show='headings', yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
    tree.heading('Original', text='原文件名')
    tree.heading('Translated', text='翻译文件名')
    tree.heading('Open', text='')

    # 配置滚动条
    tree_scrollbar_y.config(command=tree.yview)
    tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    tree_scrollbar_x.config(command=tree.xview)
    tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 插入数据
    def insert_data(file_list):
        # 清空表格
        for item in tree.get_children():
            tree.delete(item)
        for idx, (original, translated) in enumerate(file_list):
            tree.insert('', 'end', iid=idx, values=(original, translated, 'Open'))

    # 初始插入所有数据
    insert_data(file_pairs)

    # 绑定Open按钮点击事件
    def on_open_click(event):
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if item and column == '#3':  # '#3' 是 'Open' 列
            original_name = tree.item(item, 'values')[0]
            open_file(root_dir, original_name)

    tree.bind('<Button-1>', on_open_click)

    # 过滤功能
    def on_filter_change(*args):
        filter_text = filter_entry.get()
        filtered_files = []
        for original, translated in file_pairs:
            if filter_text in original or filter_text in translated:
                filtered_files.append((original, translated))
        insert_data(filtered_files)

    filter_entry.bind('<KeyRelease>', on_filter_change)

    root.mainloop()

# 主函数
if __name__ == '__main__':
    txt_file = 'file_list.txt'  # 请将此处的文件名替换为您的实际文件名
    root_dir, file_pairs = read_file_list(txt_file)
    create_gui(root_dir, file_pairs)
