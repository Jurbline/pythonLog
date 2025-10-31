# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月31日22时39分49秒
# 下午10:39


my_list = "This is a test string from Andrew".split()
print(my_list)

def change_lower(str_name:str):
    return str_name.lower()
print(sorted(my_list, key=change_lower))