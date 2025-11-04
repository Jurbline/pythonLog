# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月31日22时39分49秒
# 下午10:39


my_list = "This is a test string from Andrew".split()
print(my_list)


def change_lower(str_name: str):
    return str_name.lower()


print(sorted(my_list, key=change_lower))
print(my_list)

print('-' * 50)
student_tuples = [
    ('jane', 'B', 12),
    ('john', 'A', 15),
    ('dave', 'B', 10),
]

# use lambda func
print(sorted(student_tuples, key=lambda x: x[1]))


class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age

    def __repr__(self):
        return repr((self.name, self.grade, self.age))


student = Student('john', 'A', 15)
student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]
print('-' * 50)
print(sorted(student_objects, key=lambda student: student.age))

from operator import itemgetter, attrgetter

print("\n========== use operator ==========\n")
print(sorted(student_tuples, key=itemgetter(1)))
print(sorted(student_objects, key=attrgetter('age')))

print("\n========== mix operator ==========\n")
print(sorted(student_tuples, key=itemgetter(1, 2)))
print(sorted(student_objects, key=attrgetter('grade', 'age')))
print(sorted(student_tuples, key=lambda x: (x[1], -x[2])))

print("Check sorting stability")
data = [('red', 1), ('blue', 1), ('red', 2), ('blue', 2)]
print(sorted(data, key=itemgetter(0)))

mydict = {'Li': ['M', 7],
          'Zhang': ['E', 2],
          'Wang': ['P', 3],
          'Du': ['C', 2],
          'Ma': ['C', 9],
          'Zhe': ['H', 7]}

print(sorted(mydict.items(), key=lambda x: x[1][1]))

gameresult = [
    {"name": "Bob", "wins": 10, "losses": 3, "rating": 75.00},
    {"name": "David", "wins": 3, "losses": 5, "rating": 57.00},
    {"name": "Carol", "wins": 4, "losses": 5, "rating": 57.00},
    {"name": "Patty", "wins": 9, "losses": 3, "rating": 71.48}]
print('-' * 50)
print(sorted(gameresult, key=lambda x: x['rating']))
print(sorted(gameresult, key=itemgetter("rating", "name")))
