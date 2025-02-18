import os


def get_files_content(directory, ignore_files=None, output_file='output.txt', extensions=None):
    # 如果没有指定忽略的文件，默认是空列表
    if ignore_files is None:
        ignore_files = []

    # 如果没有指定文件扩展名，默认处理 '.py' 和 '.html'
    if extensions is None:
        extensions = ['.py', '.html']

    with open(output_file, 'w', encoding='utf-8') as output:
        for root, _, files in os.walk(directory):
            for file in files:
                # 如果文件名在忽略列表中，则跳过
                if file in ignore_files:
                    continue

                # 检查文件扩展名是否是我们想要的类型
                if not any(file.endswith(ext) for ext in extensions):
                    continue

                file_path = os.path.join(root, file)

                # 只处理文件，不处理文件夹
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            output.write(f"--- Content of {file_path} ---\n")
                            output.write(content)
                            output.write("\n\n")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    print(f"Contents have been written to {output_file}")


# 示例使用：
directory_path = './src'  # 需要扫描的文件夹路径
ignore_files = ['.env', 'ignore_file2.txt']  # 需要忽略的文件列表
extensions = ['.py', '.css', '.js', '.html']  # 只处理 .py 和 .html 后缀的文件

get_files_content(directory_path, ignore_files, 'output.txt', extensions)
