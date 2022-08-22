class Node(object):
    """单链表的结点"""

    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class SingleLinkList(object):
    """带头尾节点的单链表"""

    def __init__(self, item):
        self._head = None
        self._end = None
        node = Node(item)
        node.next = self._head
        self._head = node
        self._end = node

    def is_empty(self):
        """判断链表是否为空"""
        return self._head is None

    def empty(self):
        """链表清空"""
        self._head = None
        self._end = None

    def items(self):
        """遍历链表"""
        # 获取head指针
        cur = self._head
        # 循环遍历
        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next

    def append_list(self, node_head, node_end):
        """添加链表，通过头尾相连，并更新尾节点位置，以达到链表衔接"""
        if self.is_empty():
            # 空链表，_head 指向新结点
            self._head = node_head
            self._end = node_end
        elif node_head is not None:
            # 不是空链表，完成添加
            self._end.next = node_head
            self._end = node_end

    def head(self):
        """返回该链表的头节点"""
        return self._head

    def end(self):
        """返回该链表的尾节点"""
        return self._end
