# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月26日22时23分54秒
# 下午10:23
def read_conf():
    """
    read config
    :return:
    """
    file = open('config', 'r+', encoding='utf8')
    text_info = file.read()
    print(text_info)
    my_dict = eval(text_info)
    print(f"text_info type:{type(text_info)}")
    print(f"my_dict type:{type(my_dict)}")
    file.close()


if __name__ == '__main__':
    # read_conf()
    # eval("__import__('os').system('ls')")