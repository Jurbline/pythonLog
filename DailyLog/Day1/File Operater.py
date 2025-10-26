# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月26日21时47分13秒
# 下午9:47
def open_r():
    """
    read the file
    :return: 
    """
    file = open("file1", mode="r", encoding="utf8")
    text = file.read()
    print(text)
    file.close()


def open_rw():
    file = open("file1", mode="r+", encoding="utf8")
    text = file.read()
    print(text)
    file.write('World')  # End
    file.close()


def open_w():
    file = open('file3', mode="w", encoding='utf8')
    # file.write("Hello,我爱学习")
    file.close()


def open_a():
    file = open('file1', mode='a', encoding="utf8")
    file.write('how')
    file.close()


def use_readline():
    file = open('file2.txt', encoding="utf8")

    while True:
        text = file.readline()

        if not text:
            break

        print(text, end="")

    file.close()


if __name__ == '__main__':
    # open_r()
    # open_rw()
    # open_w()
    # open_a()
    use_readline()