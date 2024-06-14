# -*- coding: UTF-8 -*-

# 本工具用于自动增加CoreMinimal.h到头文件中，保持IWYU约定
# 顺带将非UTF8编码文件转为UTF8编码文件，避免部分文本编辑器打开看到汉字乱码

#------------------------------------------------
# 文件类型配置区域
#------------------------------------------------
# 不要处理的文件列表
exclude_files = ["pb.h"]

# 除了.h文件外要检测进行UTF8编码修正的文件类型
fix_non_utf8_file_types = ['.cpp']

# 要转为UTF8编码的
fix_codecs = ['UTF-8-SIG', 'GB2312']


# 判定编码是否需要进行修正
def should_fix(codec):
    for c in fix_codecs:
        if codec == c:
            print("Should fix codec:" + codec)
            return True
    
    return False


#------------------------------------------------
# 运行参数区域
#------------------------------------------------
# 默认仅预览修改列表，不进行文件替换保存
preview_run = False

# 自动添加CoreMinimal.h
auto_add_coreminimal_h = True

# 自动将文件转为utf-8编码
auto_convert_to_utf8 = True

# 自动打开替换过的文件，建议用notepad++可以方便查看编码
notepad_app = "notepad++.exe"

# 自动用notepad_app打开修改的文件
auto_open_changed = False

# 最终统一打印出来的错误信息
error_log_text = ""


# 处理目录中所有的.h和.cpp文件
def proc_dir(dir_name):
    import os
    dir_name = os.path.abspath(dir_name)
    print(action_type() + "Processing " + dir_name)
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".h"):
                fix_include_file(file_path)
            else:
                for ext in fix_non_utf8_file_types:
                    if file.endswith(ext):
                        fix_utf8_file(file_path)



# LOG用的提示文字当前为预览还是正式执行
def action_type():
    if preview_run:
        return "[预览模式]"
    else:
        return "[执行]"



# 自动用notepad_app打开修改的文件
def open_to_edit(file_name):
    if auto_open_changed:
        import subprocess
        cmd_str = notepad_app + " " + file_name
        # print(cmd_str)
        subprocess.Popen(cmd_str, shell=True)



# 修正头文件
def fix_include_file(file_name):
    for exclude in exclude_files:
        if file_name.endswith(exclude):
            print(action_type() + "Skip exclude file:" + exclude)
            return

    #if not file_name.endswith("ESCharacterAnimAssetListBase.h"):
    #    return

    import io
    import chardet

    with io.open(file_name, "rb") as file:
        file_data = file.read()
        file.close()

        code_detect = chardet.detect(file_data)
        codec = code_detect["encoding"]
        #print("检测编码：", codec)

        import codecs
        data = file_data.decode(codec)

        found_minimal = data.find("CoreMinimal.h")
        if found_minimal == -1:
            auto_add_include(codec, data, file_name)
        elif should_fix(codec):
            convert_h_to_utf8(codec, data, file_name)



# 将文件存为UTF8编码
def save_file_utf8(codec, data, file_name):
    if preview_run:
        return

    import codecs
    with codecs.open(file_name, "w", 'utf-8') as file:    
        file.write(data)
    
    print(action_type() + "!!! 文件已经保存 codec:{0} file_name:{1}".format(codec, file_name))



# 转换头文件为UTF8编码
def convert_h_to_utf8(codec, data, file_name):
    if not auto_convert_to_utf8:
        return

    print(action_type() + ">>> 转为UTF8文件 codec:{0}    file_name:{1}".format(codec, file_name))
    open_to_edit(file_name)
    save_file_utf8(codec, data, file_name)



# 自动添加CoreMinimal.h头文件，并添加没有#pragma once的
def auto_add_include(codec, data, file_name):
    if not auto_add_coreminimal_h:
        return
    
    add_pragma_once = data.find("#pragma once") == -1
    print(action_type() + "+++ 自动添加 CoreMinimal.h codec:{0}  add_pragma_once:{1} file_name:{2}".format(
        codec, add_pragma_once, file_name))

    open_to_edit(file_name)

    # 如果不是预览模式就自动执行文件写入
    if not preview_run:
        core_minmal_str = "#include \"CoreMinimal.h\"\r\n"
        replace_target_str = "#pragma once\r\n" + core_minmal_str

        # 没有pragma once的自动添加
        if add_pragma_once:
            data = replace_target_str + data
        else:
            data = data.replace("#pragma once", replace_target_str)

        save_file_utf8(codec, data, file_name)


# 自动修正非UTF8编码的文件
def fix_utf8_file(file_name):
    if not auto_convert_to_utf8:
        return

    import io
    import chardet

    with io.open(file_name, "rb") as file:
        file_data = file.read()
        #file.close()
        code_detect = chardet.detect(file_data)
        codec = code_detect["encoding"]
        if codec == None:
            log_str = "[Error]文件为空：" + file_name + "\r\n"
            print(log_str)

            global error_log_text
            error_log_text = error_log_text + log_str            
            return

        # print("检测编码：", codec, file_name)

        import codecs
        data = file_data.decode(codec)

        if should_fix(codec):
            print(action_type() + ">>> 转为UTF8文件 codec:{0}    file_name:{1}".format(codec, file_name))
            open_to_edit(file_name)
            save_file_utf8(codec, data, file_name)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import sys

    arg_len = len(sys.argv)

    for arg in sys.argv[1:]:
        print(arg)

    proc_dir("../../Source")
    # proc_dir("../Plugins")

    print("================处理结束================")
    print(error_log_text)
