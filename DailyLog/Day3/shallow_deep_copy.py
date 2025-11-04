# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年11月04日15时34分00秒
# 下午3:34

import copy


def use_list_copy():
    """
    use the list copy
    :return:
    """
    a = [1, 2, 3]
    b = a.copy()
    b[0] = 0
    print(id(a))
    print(id(b))
    print(a)
    print(b)


def use_copy():
    c = [1, 2, 3]
    d = copy.copy(c)
    d[1] = 5
    print(id(c))
    print(id(d))
    print(c)
    print(d)


def use_copy2():
    a = [1, 2]
    b = [3, 4]
    c = [a, b]
    d = copy.copy(c)
    print(f'c-id = {id(c)}')
    print(f'd-id = {id(d)}')
    print('-' * 50)
    print(id(c[0]), id(c[1]))
    print(id(d[0]), id(d[1]))


def use_deepcopy():
    a = [1, 2]
    b = [3, 4]
    c = [a, b]
    d = copy.deepcopy(c)
    a[0] = 666
    print(f'c-id = {id(c)}')
    print(f'd-id = {id(d)}')
    print(c)
    print(d)
    print('-' * 50)
    print(id(c[0]), id(c[1]))
    print(id(d[0]), id(d[1]))


class Hero:
    def __init__(self, name, blood):
        self.name = name
        self.blood = blood
        self.equipment = ['Shoes', 'Sword']


def use_copy_own_obj():
    old_hero = Hero('Ant', 100)
    new_hero = copy.deepcopy(old_hero)
    new_hero.blood = 80
    new_hero.equipment.append('Helmet')
    print(old_hero.blood)
    print(old_hero.equipment)
    print(new_hero.blood)
    print(new_hero.equipment)


if __name__ == '__main__':
    # use_list_copy()
    # use_copy()
    # use_copy2()
    # use_deepcopy()
    use_copy_own_obj()
