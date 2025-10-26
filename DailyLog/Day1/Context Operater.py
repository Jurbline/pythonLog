# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月26日22时10分42秒
# 下午10:10
import os


def use_rename():
    # os.rename('dir1/file1','dir1/file2')
    os.remove('dir1/file2')


def use_dir_func():
    file_list = os.listdir('.')
    print(file_list)
    # os.mkdir('dir2')
    # os.rmdir('dir2')
    print(os.getcwd())
    os.chdir('dir1')
    file = open('file1', 'w', encoding='utf8')
    file.close()


def change_dir():
    print(os.getcwd())
    os.chdir('dir1')
    print(os.getcwd())


def scan_dir(current_path, width):
    file_list = os.listdir(current_path)
    for file in file_list:
        print(' ' * width, file)
        new_path = current_path + '/' + file
        if os.path.isdir(new_path):
            scan_dir(new_path, width + 4)


def use_stat(file_path):
    file_info = os.stat(file_path)
    print('size{},uid{},mode{:x},mtime{}'.format(file_info.st_size, file_info.st_uid,
                                                 file_info.st_mode, file_info.st_mtime))
    from time import strftime
    from time import gmtime
    gm_time = gmtime(file_info.st_mtime)
    print(strftime("%Y-%m-%d %H:%M:%S", gm_time))

if __name__ == '__main__':
    # use_rename()
    # use_dir_func(
    # change_dir()
    # scan_dir('.', 0)
    use_stat('file1')