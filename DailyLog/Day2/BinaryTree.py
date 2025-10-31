# Python by PyCharm
# This is a Head Script
# Editor:Sueyuer
# 2025年10月31日10时33分37秒
# 上午10:33
class Node:
    def __init__(self, elem = -1, lchild = None, rchild = None):
        self.elem =elem
        self.lchild = lchild
        self.rchild = rchild

class BinaryTree:
    def __init__(self):
        self.root = None
        self.help_queue = []

    def build_tree(self,node:Node):
        if self.root is None:
            self.root = node
            self.help_queue.append(node)

        else:
            self.help_queue.append(node)
            if self.help_queue[0].lchild is None:
                self.help_queue[0].lchild = node
            else:
                self.help_queue[0].rchild = node
                self.help_queue.pop(0)

    def pre_order(self,current_node:Node):
        if current_node:
            print(current_node.elem, end=' ')
            self.pre_order(current_node.lchild)
            self.pre_order(current_node.rchild)

    def mid_order(self,current_node:Node):
        if current_node:
            self.mid_order(current_node.lchild)
            print(current_node.elem, end=' ')
            self.mid_order(current_node.rchild)

    def last_order(self,current_node:Node):
        if current_node:
            self.last_order(current_node.lchild)
            self.last_order(current_node.rchild)
            print(current_node.elem, end=' ')

    def level_order(self,root_node:Node):
        help_queue = []
        help_queue.append(root_node)
        while help_queue:
            out_node = help_queue.pop(0)
            print(out_node.elem, end=' ')
            if out_node.lchild:
                help_queue.append(out_node.lchild)
            if out_node.rchild:
                help_queue.append(out_node.rchild)

if __name__ == '__main__':
    tree = BinaryTree()
    for i in range(1,11):
        new_node = Node(i)
        tree.build_tree(new_node)
    tree.pre_order(tree.root)
    print()
    tree.mid_order(tree.root)
    print()
    tree.last_order(tree.root)
    print()
    tree.level_order(tree.root)