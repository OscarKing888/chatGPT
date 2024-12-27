import re
import sys
import os

def remove_comments(content: str) -> str:
    """移除 /*...*/ 多行注释及 //... 单行注释"""
    # DOTALL 让 '.*?' 可以跨行匹配
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//.*', '', content)
    return content

def get_param_names(param_string: str) -> str:
    """
    从形参串中提取每个参数的变量名（去掉类型、默认值），并用逗号拼回去。
    比如 "const FString& MapName, int32 Count = 10" -> "MapName, Count"
    """
    param_string = param_string.strip()
    if not param_string:
        return ""

    # 拆分逗号
    params = param_string.split(',')
    var_names = []
    # 使用一个正则去匹配每个参数的 [类型(可含模板/指针引用)] + [变量名] + [默认值]
    # 大致形如:  "const FString& MapName" / "int32 Count = 10" / "TSubclassOf<AGameSession> SessionClass"
    # group(1) 就是变量名
    pattern = re.compile(r'''
        ^                       # 开头
        (?:[\w:\<>\(\)\&\*\s]+) # 类型部分(包含可能的模板、引用、指针等)
        \s+                     # 至少一个空格
        ([A-Za-z_]\w*)         # 变量名(捕获)
        (?:\s*=\s*.*)?         # 可选的 = 默认值
        $                       # 结束
    ''', re.VERBOSE)

    for p in params:
        p = p.strip()
        m = pattern.match(p)
        if m:
            var_names.append(m.group(1))
        else:
            # 如果没匹配上，就原样放回（或可视为错误处理）
            var_names.append(p)

    return ", ".join(var_names)

def extract_virtual_functions(header_path: str):
    """
    从 .h 文件中提取带 virtual 的单行函数声明：
    1. 去除注释后，逐行判断是否同时含 'virtual' + 以 ';' 结尾
    2. 去掉 [A-Z_]+_API 宏
    3. 在末尾补上 'override;'
    4. 返回提取好的行列表
    """
    if not os.path.isfile(header_path):
        print(f"文件不存在: {header_path}")
        return []

    with open(header_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 去掉注释
    content = remove_comments(content)

    lines = content.split('\n')
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 跳过常见 UE 宏行
        if ('UCLASS(' in line 
            or 'GENERATED_' in line
            or 'UPROPERTY(' in line
            or 'UMETA(' in line
            or 'meta=' in line
            or 'UFUNCTION(' in line):
            continue

        # 条件：含有 'virtual' 且以分号结尾
        if 'virtual' in line and line.endswith(';'):
            # 去掉类似 ENGINE_API / CORE_API / XXX_API
            line = re.sub(r'\b[A-Z_]+_API\b\s*', '', line)
            # 去掉多余空白
            line = line.strip()
            # 若没有 override;，就补上
            if not line.endswith('override;'):
                line = line.rstrip(';') + ' override;'
            results.append(line)

    return results

def parse_function_signature(signature: str):
    """
    将类似：
    "virtual TSubclassOf<AGameSession> GetGameSessionClass() const override;"
    拆分为若干字段：返回类型 / 函数名 / 参数列表 / 是否 const
    返回 (return_type, function_name, params, is_const)
    """
    # 简单正则：virtual + (返回类型) + (函数名) + (参数) + [可选 const] + override;
    pattern = re.compile(
        r'^virtual\s+(?P<ret>[\w:\<>\(\)\s\*&]+)\s+'    # 返回类型(可能含模板、指针、引用等)
        r'(?P<name>[\w:\~]+)\s*'                        # 函数名(含可能的 ~析构 之类)
        r'\((?P<params>[^)]*)\)\s*'                     # 参数列表(不含括号)
        r'(?P<const>const\s*)?'                         # 可选的 const
        r'override;$'                                   # 末尾 override;
    )

    m = pattern.match(signature.strip())
    if not m:
        return None
    
    ret_type   = m.group('ret').strip()
    func_name  = m.group('name').strip()
    params     = m.group('params').strip()
    is_const   = (m.group('const') is not None)
    return (ret_type, func_name, params, is_const)

def generate_cpp_definitions(class_name: str, signatures: list[str]) -> list[str]:
    """
    根据提取的函数声明列表，生成 .cpp 中的实现代码。
    每个函数实现形如：
        返回类型 A<class_name>::函数名(形参) [const?]
        {
            // 如果返回类型 != void
            //   auto RetVal = Super::函数名(形参变量名);
            //   LOG_FUN();
            //   return RetVal;
            // 否则
            //   Super::函数名(形参变量名);
            //   LOG_FUN();
        }
    """
    cpp_lines = []
    for sig in signatures:
        parsed = parse_function_signature(sig)
        if not parsed:
            # 如果没法解析，跳过
            continue

        ret_type, func_name, params, is_const = parsed
        # 类名加个 A 前缀，以示是 Actor 类 (你可根据项目情况自行调整)
        full_class_name = f"A{class_name}"  
        # 生成函数头
        const_str = " const" if is_const else ""

        # 去除 virtual/override 之后的最终声明
        func_declaration = f"{ret_type} {full_class_name}::{func_name}({params}){const_str}"
        cpp_lines.append(func_declaration)
        cpp_lines.append("{")

        # 解析出仅变量名，用于 Super:: 调用
        var_names = get_param_names(params)

        if ret_type != "void":
            cpp_lines.append(f"    auto RetVal = Super::{func_name}({var_names});")
            cpp_lines.append("    LOG_FUN();")
            cpp_lines.append("    return RetVal;")
        else:
            cpp_lines.append(f"    Super::{func_name}({var_names});")
            cpp_lines.append("    LOG_FUN();")

        cpp_lines.append("}")
        cpp_lines.append("")  # 空行分隔

    return cpp_lines

def main():
    if len(sys.argv) < 2:
        print("用法: python script.py <HeaderFile.h>")
        print("示例: python script.py GameModeBase.h")
        sys.exit(1)

    header_file = sys.argv[1]
    if not os.path.isfile(header_file):
        print(f"找不到文件: {header_file}")
        sys.exit(1)

    # 获取不带后缀的文件名，用来生成类名、输出文件名
    base_name = os.path.splitext(os.path.basename(header_file))[0]  # 例如 "GameModeBase"
    
    # 提取所有带 virtual 的函数声明
    virtual_signatures = extract_virtual_functions(header_file)
    if not virtual_signatures:
        print("没有找到任何 virtual 函数声明。")
        return

    # 生成新的 .h 文件名 和 .cpp 文件名（加个"_Override"后缀避免覆盖）
    new_header_name = f"{base_name}_Override.h"
    new_cpp_name    = f"{base_name}_Override.cpp"

    # 1) 写出新的 .h 文件：只需要把提取到的签名写进去就行
    with open(new_header_name, 'w', encoding='utf-8') as f:
        f.write(f"#pragma once\n\n")
        f.write(f"// Generated override header for {header_file}\n\n")
        f.write(f"class A{base_name}\n")
        f.write("{\npublic:\n\n")
        for sig in virtual_signatures:
            f.write(f"    {sig}\n")
        f.write("};\n")

    # 2) 生成对应的 .cpp 内容
    cpp_lines = []
    cpp_lines.append(f'#include "{new_header_name}"')
    cpp_lines.append("")
    cpp_lines.append(f"// Generated override cpp for {header_file}")
    cpp_lines.append("")

    # 生成每个函数的定义（Super:: 函数调用 + LOG_FUN();）
    impl_lines = generate_cpp_definitions(base_name, virtual_signatures)
    cpp_lines.extend(impl_lines)

    # 写出新的 .cpp 文件
    with open(new_cpp_name, 'w', encoding='utf-8') as f:
        for line in cpp_lines:
            f.write(line + "\n")

    print(f"已在当前目录生成文件: {new_header_name} 和 {new_cpp_name}")


if __name__ == "__main__":
    main()
