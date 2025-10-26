# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月26日22时21分35秒
# 下午10:21
import sys

# print(type(sys.argv))
# print(sys.argv)

def write_hello(file_path):
    file = open(file_path,'w+',encoding='utf8')
    file.write('hello')
    file.close()

if __name__ == '__main__':
    write_hello(sys.argv[1])