import os
import sys

def rename_items_in_directory(directory_path, old_text, new_text):
    """
    递归遍历目录，重命名包含指定文本的文件和文件夹
    """
    if not os.path.exists(directory_path):
        print(f"错误: 目录 {directory_path} 不存在")
        return
    
    renamed_count = 0
    
    # 注意：我们需要从最深层开始重命名，避免路径变化影响
    # 使用 os.walk 的 topdown=False 参数，从子目录开始处理
    for root, dirs, files in os.walk(directory_path, topdown=False):
        # 先处理文件
        for file in files:
            if old_text in file:
                old_path = os.path.join(root, file)
                new_file_name = file.replace(old_text, new_text)
                new_path = os.path.join(root, new_file_name)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"重命名文件: {old_path} -> {new_path}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名文件失败: {old_path} -> {new_path} - {str(e)}")
        
        # 再处理目录（从最深层开始，所以不会影响父目录路径）
        for dir_name in dirs:
            if old_text in dir_name:
                old_path = os.path.join(root, dir_name)
                new_dir_name = dir_name.replace(old_text, new_text)
                new_path = os.path.join(root, new_dir_name)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"重命名目录: {old_path} -> {new_path}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名目录失败: {old_path} -> {new_path} - {str(e)}")
    
    return renamed_count

def main():
    directory_path = "/home/fengxiaoyao/1023FilterVector"
    
    old_text = "ACORN-gamma-improve"
    new_text = "ACORN-gamma-improved"
    
    print(f"开始在目录 '{directory_path}' 中查找并重命名包含 '{old_text}' 的文件和文件夹为 '{new_text}'")
    print("注意: 此操作会直接重命名文件和文件夹，请确保已备份重要数据！")
    
    confirm = input("是否继续？(y/N): ").strip().lower()
    if confirm != 'y' and confirm != 'yes':
        print("操作已取消")
        return
    
    renamed_count = rename_items_in_directory(directory_path, old_text, new_text)
    print(f"\n处理完成! 共重命名了 {renamed_count} 个文件/文件夹")

if __name__ == "__main__":
    main()