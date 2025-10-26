# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月26日21时32分41秒
# 下午9:32
class A:
    def __init__(self):
        self.__age = 18
    def show_age(self):
        print(self.__age)

class B(A):
    def get_age(self):
        self.show_age()

if __name__ == '__main__':
    Sample = B()
    Sample.get_age()