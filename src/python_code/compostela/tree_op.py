from graphviz import Digraph
import matplotlib.pyplot as plt


class Tree_op:
    """
    Tree auxiliar operations
    """

    @staticmethod
    def plot_tree_plt(tree):
        """
        Plot the tree structure in a matplotlib graph
        :param tree: tree structure
        :return:
        """

        def print_points(j, tree):
            """
            Scatter the nodes in matplotlib
            :param j: node
            :param tree: tree structure
            :return:
            """
            plt.scatter(tree[j]["point"][0], tree[j]["point"][1],
                        label=str(j))
            if tree[j]["children"] != []:
                for i in tree[j]["children"]:
                    print_points(i, tree)


        def print_plot(j, tree):
            """
            Plot edges in matplotlib
            :param j:node
            :param tree: tree structure
            :return:
            """
            if tree[j]["children"] != []:
                for i in tree[j]["children"]:
                    plt.plot([tree[j]["point"][0],
                              tree[i]["point"][0]],
                             [tree[j]["point"][1],
                              tree[i]["point"][1]], "b-")
                    print_plot(i, tree)

        print_plot(0, tree)  # draw nodes
        print_points(0, tree)  # draw edges

    @staticmethod
    def print_tree(j, tree, space=""):
        """
        Print tree structure in the terminal
        :param j: node to start printing
        :param tree: tree structure
        :param space: space string
        :return:
        """
        print(space + "Node {}:".format(j), tree[j]["point"])
        if tree[j]["children"] != []:
            space = space + "   "
            for i in tree[j]["children"]:
                Tree_op.print_tree(i, tree, space=space)

    @staticmethod
    def plot_tree(tree, imgs, num1, num2, path="save_candidates/",
                  path_tree="trees/", name="Compostela Tree"):
        """
        Plot tree using graphviz
        :param tree: tree structure
        :param imgs: candidates image array
        :param path: path to save images
        :param path_tree: path to save tree image
        :param name: Name render tree
        :return:
        """
        def nodes(father, node_list):
            """
            Render nodes tree
            :param father: father ndoe
            :param node_list: list of nodes with edges
            :return:
            """
            with dot.subgraph() as s:  # Need it to create a tree
                s.attr(rank='same')
                if tree[father]["children"] != []:
                    for i in tree[father]["children"]:
                        plt.figure(str(i) + " img")
                        plt.imshow(imgs[i], cmap='gray')
                        plt.savefig(path + str(i) + ".png")
                        dot.node(str(i),
                                 image= str(i) + ".png",
                                 xlabel="Node: " + str(i))
                        node_list.append([str(father), str(i)])
                        node_list = nodes(i, node_list)
            return node_list

        dot = Digraph(comment=name)  # Initialize graph
        plt.figure(0)
        plt.imshow(imgs[0], cmap='gray')
        plt.savefig(path + str(0) + ".png")
        dot.node(str(0), image= str(0) + ".png",
                 xlabel="Node: " + str(0))
        node_list = nodes(0, [])  # Render nodes
        for i in node_list:  # Render edges
            dot.edge(i[0], i[1])
        # Save
        dot.render(path + "tree")

