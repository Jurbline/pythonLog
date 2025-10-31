# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月31日11时01分03秒
# 上午11:01
import random
import time
import sys

sys.setrecursionlimit(1000000)


class Sort:
    def __init__(self, n):
        self.len = n
        # self.arr = [3, 87, 2, 93, 78, 56, 61, 38, 12, 40]
        self.arr = [0] * n
        self.random_data()

    def random_data(self):
        for i in range(self.len):
            self.arr[i] = random.randint(0, 10)

    def partition(self, left, right):
        arr = self.arr
        k = i = left
        random_pos = random.randint(left, right)  # 每次在里面随机挑一个值作为基准
        arr[random_pos], arr[right] = arr[right], arr[random_pos]  # 避免陷入最坏的时间复杂度
        for i in range(left, right):
            if arr[i] < arr[right]:
                arr[i], arr[k] = arr[k], arr[i]
                k += 1
        arr[k], arr[right] = arr[right], arr[k]
        return k

    def quick_sort(self, left, right):
        if left < right:
            pivot = self.partition(left, right)
            self.quick_sort(left, pivot - 1)
            self.quick_sort(pivot + 1, right)

    def adjust_max_heap(self, pos, arr_len):
        """
        :param pos: 被调整的元素的位置（父亲）
        :param arr_len: 当时的列表总长度
        :return:
        """
        arr = self.arr
        parent = pos
        son = 2 * parent + 1
        while son < arr_len:  # 如果是while True的话，最上面的元素到最下面的话就会多出来一个结点了，因此结束条件不能超出数组长度
            if son + 1 < arr_len and arr[son] < arr[son + 1]:  # 右孩子存在且右孩子大于左孩子
                son += 1  # 这里的son是要跟parent交换的元素
            if arr[son] > arr[parent]:
                arr[son], arr[parent] = arr[parent], arr[son]
                parent = son
                son = 2 * parent + 1
            else:  # 当前子树是满足大根堆的
                break

    def heap_sort(self):
        # 把列表调整为大根堆
        for parent in range(self.len // 2 - 1, -1, -1):  # 从中间向前开始遍历
            self.adjust_max_heap(parent, self.len)  # 这一轮之后，就筛选出了最大的那个元素（堆顶），把它跟堆底的元素交换（继续堆排）
        arr = self.arr
        arr[0], arr[self.len - 1] = arr[self.len - 1], arr[0]
        for arr_len in range(self.len - 1, 1, -1):  # 已经筛选出一个了，这次只需要筛选 arr_len - 1 个，1是因为2个元素才有排序(0,1)
            self.adjust_max_heap(0, arr_len)  # 因为调整到堆顶，这时候堆顶就不满足了
            arr[0], arr[arr_len - 1] = arr[arr_len - 1], arr[0]

    def statistical_time(self, sort_func, *args, **kwargs):
        """
        回调函数
        :param sort_func:
        :param args:
        :param kwargs:
        :return:
        """
        start = time.time()
        sort_func(*args, **kwargs)
        end = time.time()
        print(f"总计用时{end - start}")


if __name__ == '__main__':
    my_sort = Sort(100000)
    # my_sort.quick_sort(0, my_sort.len - 1)
    # print(my_sort.arr)
    # my_sort.heap_sort()
    # my_sort.statistical_time(my_sort.quick_sort, 0, my_sort.len - 1)
    my_sort.statistical_time(my_sort.heap_sort)
    # print(my_sort.arr)
