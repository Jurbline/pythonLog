# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月31日11时01分03秒
# 上午11:01

class Sort:
    def __init__(self, n):
        self.len = n
        self.arr = [3, 87, 2, 93, 78, 56, 61, 38, 12, 40]

    def partition(self, left, right):
        arr = self.arr
        k = i = left
        for i in range(left, right):
            if arr[i] < arr[right]:
                arr[i],arr[k]=arr[k],arr[i]
                k += 1
        arr[k],arr[right]=arr[right],arr[k]
        return k


    def quick_sort(self, left, right):
        if left < right:
            pivot = self.partition(left, right)
            self.quick_sort(left,pivot - 1)
            self.quick_sort(pivot + 1,right)

if __name__ == '__main__':
    my_sort = Sort(10)
    my_sort.partition(0, my_sort.len-1)
    my_sort.quick_sort(0, my_sort.len - 1)
    pass