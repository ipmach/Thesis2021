from abc import ABC, abstractmethod
from Visualizer.render_pygame import pygame_render


class Figure_basic(ABC):

    def initBoard(self, ttt, number_lines=20):
        """
        Initialize the board
        :param ttt: windows instance
        :param number_lines: Number of lines to draw
        :return: background of board
        """
        # set up the background surface
        self.x_0, self.y_0 = int(self.width/2.2), int(self.height/1.8)
        self.x_max, self.y_max = self.width - 100, self.height - 100
        self.x_min, self.y_min = 100, 100
        background = self.render.create_surface(ttt)
        space_x = (self.y_max - self.x_0) / number_lines
        space_y = (self.x_max - self.y_0) / number_lines
        # Draw lines
        for i in range(1, number_lines):
            self.render.draw_line(background, (220,220,220),
                                 [(self.x_max, self.x_0 - space_x * i),
                                 (self.x_min, self.x_0 - space_x * i)], 1)
            self.render.draw_line(background, (220,220,220),
                                 [(self.x_max, self.x_0 + space_x * i),
                                 (self.x_min, self.x_0 + space_x * i)], 1)
            self.render.draw_line(background, (220,220,220),
                                 [(self.y_0 - space_y * i, self.y_max),
                                 (self.y_0 - space_y * i, self.y_min)], 1)
            self.render.draw_line(background, (220,220,220),
                                 [(self.y_0 + space_y * i, self.y_max),
                                 (self.y_0 + space_y * i, self.y_min)], 1)
        # Draw center lines
        self.render.draw_line(background, (0,0,0),
                             [(self.x_max, self.x_0),
                             (self.x_min, self.x_0)], 3)
        self.render.draw_line(background, (0,0,0),
                             [(self.y_0, self.y_max),
                             (self.y_0, self.y_min)], 3)
        # Draw borders
        self.render.draw_line(background, (0,128,128),
                             [(self.x_min, self.y_max),
                             (self.x_max, self.y_max)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(self.x_min, self.y_min),
                             (self.x_max, self.y_min)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(self.x_max, self.y_max),
                             (self.x_max, self.y_min)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(self.x_min, self.y_max),
                             (self.x_min, self.y_min)], 3)

        return background

    def set_x_label(self, message):
        """
         Set x label
         :param message: x label message
         :return:
         """
        self.x_label = message

    def set_y_label(self, message):
        """
        Set y label
        :param message: y label message
        :return:
        """
        self.y_label = message

    def set_title(self, message):
        """
        Set title
        :param message: title name
        :return:
        """
        self.title = message

    def showBoard (self, ttt, board):
        """
         Display the update board.
        :param ttt: instance window
        :param board: board background
        :return:
        """
        self.render.render_surface(ttt, board)
        image_path = "src/python_code/Visualizer/img/index.png"
        self.render.render_image(ttt, image_path, (15, 15))
        self.render.render_text(ttt, self.x_label,
                                (self.x_0, self.y_max + 20))
        self.render.render_text(ttt, self.y_label,
                                (self.x_min - 60, self.y_0 - 50))
        self.render.render_text(ttt, self.title,
                                (self.x_0 - 50, self.y_min - 60),
                                fontSize=40)
        if self.labels:
            self.create_labels()

    def showCoordinates(self, ttt, pos, precision=2):
        """
        Plot coordinates in the window
        :param ttt: instance window
        :param pos: (x, y)
        :param precision: Precision between coordinates (2 -> 0.001)
        :return:
        """
        x,y = pos
        if self.x_min > x or self.x_max < x or \
        self.y_min > y or self.y_max < y:
            self.on_click = False
            message = "x: None y: None       "
        else:
            self.on_click = True
            self.x, self.y = x,y  # Save good coordinates
            x = x - self.y_0  # Transform for graph
            x = x if x == 0 else round(x / (self.x_max- self.x_0), precision)
            y = -1 * (y - self.x_0)  # Transform for graph
            y = y if y == 0 else round(y / (self.y_max- self.y_0), precision)
            message = "x: {} y: {}       ".format(x,y)
        self.render.render_text(ttt, message,
                                (self.x_min - 60, self.y_max + 60),
                                fontSize=30, background='white')
        self.render.refresh()


    def draw_point(self, board, x,y, color, r, type='o'):
        """
        Draw points in the window
        :param board: background to draw in
        :param x: Coordinate x
        :param y: Coordinate y
        :param color: color point
        :param r: radio size
        :param type: shape type
        :return:
        """
        if type == 'o':
            self.render.draw_circle(board, color, (x, y), r)
        elif type == 'x':
            self.render.draw_line(board, color, [(x - r, y - r), \
                                 (x + r, y + r)], 3)
            self.render.draw_line(board, color, [(x + r, y - r), \
                                 (x - r, y + r)], 3)

    @abstractmethod
    def create_labels(self):
        """
        Print labels for the distribution
        :return:
        """
        pass

    @abstractmethod
    def plot(self, x, y, color, r, type, z=None, name_labels=None):
        """
        Plot a distribution in the graph
        :param x: Points coordinate x
        :param y: Points coordinate y
        :param color: color of the points (no labels)
        :param r: radius of the points
        :param type: type of shape of the points
        :param z: colors each label point
        :param name_labels: list nale labes
        :return:
        """
        pass

    @abstractmethod
    def show(self):
        """
        Start the render and window loop
        :return:
        """
        pass

    @abstractmethod
    def draw_tree(self, tree, draw_are=False, show_image=True):
        """
        Draw tree in the latent space
        :param tree: json file of tree
        :param draw_are: True to draw the area of each candidate
        :param show_image: True to show the image of each candidate
        :return:
        """
        pass

    def __init__(self, figsize =(900, 800), space_right=200):
        """
        Instance of the figure
        :param figsize: size figure
        :param space_right: Space in the right size of the window
        """
        self.width, self.height = figsize[0], figsize[1]
        self.render = pygame_render  # Change render here
        self.ttt = self.render.init_render(self.width + space_right,
                                           self.height, 'Plot')
        self.x_0, self.y_0 = int(self.width/2.2), int(self.height/1.8)
        self.x_max, self.y_max = self.width - 100, self.height - 100
        self.x_min, self.y_min = 100, 100
        self.x_label = "X axis"
        self.y_label = "Y axis"
        self.title = "Latent space"
