from Visualizer.graphic import Figure_latentSpace


class Figure:

    def __init__(self,  figsize =(900, 800), space=10):
        """
        Create a plot of a latent space
        :param figsize: size figure
        :param space: Space scale distribution to the border figure
        """
        self.figsize = figsize
        self.space = space
        self.new_x_label = None
        self.new_y_label = None
        self.new_title = None

    def set_x_label(self, message):
        """
        Set x label
        :param message: x label message
        :return:
        """
        self.new_x_label = message

    def set_y_label(self, message):
        """
        Set y label
        :param message: y label message
        :return:
        """
        self.new_y_label = message

    def set_title(self, message):
        """
        Set title
        :param message: title name
        :return:
        """
        self.new_title = message

    def plot_latent(self, AE, D, number_lines=20, space_right=200,
                    color=(0,0,139), r=4, type='o', tree=None,
                    draw_are=False, show_image=False):
        """
        Plot the latent space
        :param AE: AutoeEncoder
        :param D: Distribution
        :param number_lines: Number of lines drawn in the plot
        :param space_right: Size space rigth of th window
        :param color: Background color
        :param r: Radius of each point
        :param type: Type of shape for each D point
        :param show_image: Show images of the candidate tree
        :return:
        """
        # Initialize
        fig = Figure_latentSpace(AE, figsize=self.figsize,
                                     space=self.space,
                                     space_right=space_right,
                                     number_lines=number_lines)

        # Set labels and title
        if self.new_x_label is not None:
            fig.set_x_label(self.new_x_label)
        if self.new_y_label is not None:
            fig.set_y_label(self.new_y_label)
        if self.new_title is not None:
            fig.set_title(self.new_title)

        # Plot
        fig.plot(D.x, D.y, color, r=r, type=type, z=D.z,
                      name_labels=D.labels)
        if tree is not None:
            fig.draw_tree(tree, draw_are=draw_are,
                          show_image=show_image)
        fig.show()
