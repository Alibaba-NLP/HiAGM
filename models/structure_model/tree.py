#!/usr/bin/env python
# coding:utf-8


class Tree(object):
    def __init__(self, idx):
        """
        class for tree structure of hierarchical labels
        :param: idx <- Int
        self.parent: Tree
        self.children: List[Tree]
        self.num_children: int, the number of children nodes
        """
        self.idx = idx
        self.parent = None
        self.children = list()
        self.num_children = 0

    def add_child(self, child):
        """
        :param child: Tree
        """
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        """
        :return: self._size -> Int, the number of nodes in the hierarchy
        """
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        """
        :return: int, the depth of curent node in the hierarchy
        """
        # if getattr(self, '_depth'):
        #     return self._depth
        count = 0
        if self.parent is not None:
            self._depth = self.parent.depth() + 1
        else:
            self._depth = count
        return self._depth
