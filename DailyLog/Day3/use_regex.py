# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年11月04日16时09分33秒
# 下午4:09
import re


def simple():
    result = re.match("python", "python")
    if result:
        print(result.group())


def single():
    result = re.match(".", "H")
    if result:
        print(result.group())
    ret = re.match("r.t", "ret")
    if ret:
        print(ret.group())
    ret = re.match("r.t", "rat")
    if ret:
        print(ret.group())
    print('-' * 50)
    # upper lower
    ret = re.match("[Hh]", "hello world")
    if ret:
        print(ret.group())
    ret = re.match("[Hh]", "Hello world")
    if ret:
        print(ret.group())
    ret = re.match("[Hh]ello world", "Hello world")
    if ret:
        print(ret.group())
    ret = re.match("[0-9]Hello world", "8Hello world")
    if ret:
        print(ret.group())
    ret = re.match("[0-35-9]Hello world", "8Hello world")
    if ret:
        print(ret.group())
    print('-' * 50)
    # use \d
    ret = re.match(r"ChangE No.\d", "ChangE No.1 Launch successful")
    if ret:
        print(ret.group())
    ret = re.match(r"ChangE No.\d", "ChangE No.2 Launch successful")
    if ret:
        print(ret.group())
    ret = re.match(r"ChangE No.\d", "ChangE No.3 Launch successful")
    if ret:
        print(ret.group())


def more_character():
    """
    Match multiple characters
    :return: 
    """
    ret = re.match("[A-Z][a-z]*", "M")
    print(ret.group())
    ret = re.match("[A-Z][a-z]*", "MnnM")
    print(ret.group())
    ret = re.match("[A-Z][a-z]*", "Abcdefg")
    print(ret.group())
    print('-' * 50)
    names = ["name1", "_name", "2_name", "__name__"]
    for name in names:
        ret = re.match(r"[A-Za-z_]+\w*", name)
        if ret:
            print(f"{ret.group()} is True")
        else:
            print(f"{name} is False")
    print('-' * 50)
    ret = re.match(r"[1-9]?[0-9]", "7")
    print(ret.group())

    ret = re.match(r"[1-9]?\d", "33")
    print(ret.group())

    ret = re.match(r"[1-9]?\d", "09")
    print(ret.group())

    print('-' * 50)
    ret = re.match("[a-zA-Z0-9_]{6}", "12a3g45678")
    print(ret.group())
    ret = re.match("[a-zA-Z0-9_]{8,20}", "123hj214h12k512d21dd12")
    print(ret.group())


if __name__ == '__main__':
    # simple()
    # single()
    more_character()
