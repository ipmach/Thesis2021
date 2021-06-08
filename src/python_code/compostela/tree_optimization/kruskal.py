import math
import networkx as nx
import numpy as np


class Kruskal:
    """
    Apply Kruskal to a tree structure
    """

    def get_tree(self, tree, A, a, i):
        """
        Use to convert one tree to another
        :param tree: original tree
        :param A: list of nodes
        :param a: new tree
        :param i: node number
        :return: tree, A
        """
        if A != []:
            for j in a[i]:
                if j in A:
                    tree[i]['children'].append(j)
                    tree[j]['father'] = i
                    A.remove(j)
                    tree, A = self.get_tree(tree, A, a, j)
        return tree, A

    def __call__(self, tree):
        """
        Use Kruskal
        :param tree: true structure
        :return: new tree
        """

        G = nx.Graph()  # Initialize graph
        distance = lambda x, y: math.sqrt(np.sum((x - y) ** 2))
        # Add nodes to the graph
        for i in range(len(tree)):
            tree[i]["point"] = np.array(tree[i]["point"])
            G.add_node(i)
        # Add edges to the graph
        for i in range(len(tree)):
            for j in range(i + 1, len(tree)):
                G.add_edge(i, j, weight=distance(tree[i]["point"],
                                                 tree[j]["point"]))

        # Applying minimun spanning tree
        T = nx.minimum_spanning_tree(G, algorithm="kruskal")
        a = {}
        for i, j in T.edges():
            if i not in a.keys():
                a[i] = [j]
            else:
                a[i].append(j)
            if j not in a.keys():
                a[j] = [i]
            else:
                a[j].append(i)
        i = 0
        for i in range(len(tree)):
            tree[i]['children'] = []
            tree[i]['father'] = None

        A = list(np.arange(1, len(tree)))

        tree, _ = self.get_tree(tree, A, a, 0)
        return tree