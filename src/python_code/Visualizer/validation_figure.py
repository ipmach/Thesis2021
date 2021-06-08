from Visualizer.render_pygame import pygame_render
from Visualizer.colors import range_colorbar
import numpy as np
from Visualizer.colors import colors


class validationFigure():

    def __init__(self, figsize=(1500, 950), number_lines=20):
            self.width, self.height = figsize[0], figsize[1]
            self.render = pygame_render  # Change render here
            self.ttt = self.render.init_render(self.width ,
                                               self.height, 'Plot')
            self.initBoard(self.ttt, number_lines)

            self.on_click = [False, False]

    def initBoard(self, ttt, number_lines):
        self.board = self.render.create_surface(self.ttt)
        self.total_range = 73
        self.plot_colorbar(total_range=self.total_range)
        self.plot_text_titles()
        self.x_min = [320, 820, 120, 820, 120]
        self.y_min = [120, 120, 600, 600, 120]
        self.x_max = [self.x_min[0] + 400,
                      self.x_min[1] + 400,
                      self.x_min[2] + 600,
                      self.x_min[3] + 600,
                      self.x_min[4] + 170]
        self.y_max = [self.y_min[0] + 400,
                      self.y_min[1] + 400,
                      self.y_min[2] + 300,
                      self.y_min[3] + 300,
                      self.y_min[4] + 400]
        self.x_0 = [self.x_min[0] + int((self.x_max[0] - self.x_min[0]) / 2),
                    self.x_min[1] + int((self.x_max[1] - self.x_min[1]) / 2),
                    self.x_min[2] + int((self.x_max[2] - self.x_min[2]) / 2),
                    self.x_min[3] + int((self.x_max[3] - self.x_min[3]) / 2),
                    self.x_min[4] + int((self.x_max[4] - self.x_min[4]) / 2)]
        self.y_0 = [self.y_min[0] + int((self.y_max[0] - self.y_min[0]) / 2),
                    self.y_min[1] + int((self.y_max[1] - self.y_min[1]) / 2),
                    self.y_min[2] + int((self.y_max[2] - self.y_min[2]) / 2),
                    self.y_min[3] + int((self.y_max[3] - self.y_min[3]) / 2),
                    self.y_min[4] + int((self.y_max[4] - self.y_min[4]) / 2)]
        type_plot = [[True, False, True], [True, False, True],
                     [False, True, False], [False, False, False],
                     [False, False, False]]
        # Create plots
        for i in range(len(self.x_min)):
            self.board = self.create_plot(self.board, self.y_max[i], self.x_max[i],
                                          self.y_min[i], self.x_min[i], self.y_0[i],
                                          self.x_0[i], number_lines=number_lines,
                                          draw_center=type_plot[i][0],
                                          draw_plot_lines=type_plot[i][1],
                                          draw_squares=type_plot[i][2],
                                          index=i)

    def draw_plots(self, background, index, max_value, i=10):
        max_value_y = (self.x_min[index] + 50, self.y_max[index] - 32)
        min_value_y = (self.x_max[index] - 30, self.y_max[index] - 32)
        max_value_x = (self.x_min[index] + 50, self.y_max[index] - 30)
        min_value_x = (self.x_min[index] + 50, self.y_min[index] + 30)
        # Create plot
        self.render.draw_line(background, (0, 0, 0),
                              [max_value_y, min_value_y], 5)
        self.render.draw_line(background, (0, 0, 0),
                              [max_value_x, min_value_x], 5)
        dist = max_value_y[0] - min_value_y[0]
        middle_points = (np.arange(i) + 1) * dist / (i + 1)
        dist = int((max_value_x[1] - min_value_x[1]) / 10)
        for i in range(11):
            self.render.draw_line(background, (0, 0, 0),
                                  [(max_value_x[0] - 5,
                                    max_value_x[1] - (i) * dist),
                                   (max_value_x[0] + 5,
                                    max_value_x[1] - (i) * dist)], 5)
            self.render.render_text(background,
                                    str(np.round(max_value * (i / 10), 2)),
                                    (max_value_x[0] - 39,
                                     max_value_x[1] - (i) * dist)
                                    , fontSize=20)
        for j, i in enumerate(middle_points):
            self.render.draw_line(background, (0, 0, 0),
                                  [(max_value_y[0] - i,
                                    max_value_y[1] - 5),
                                   (max_value_y[0] - i,
                                    max_value_y[1] + 5)], 5)
            self.render.render_text(background, str(j),
                                    (max_value_y[0] - i,
                                     max_value_y[1] + 15)
                                    , fontSize=20)

    def create_plot(self, background, y_max, x_max, y_min,
                  x_min, x_0, y_0, number_lines=20, draw_center=True,
                    draw_plot_lines=False, draw_squares=True, index=0):
        """
        Initialize the board
        :param ttt: windows instance
        :param number_lines: Number of lines to draw
        :return: background of board
        """
        # set up the background surface
        space_x = (y_max - x_0) / number_lines
        space_y = (x_max - y_0) / number_lines
        # Draw lines
        if draw_squares:
            for i in range(1, number_lines):
                self.render.draw_line(background, (220,220,220),
                                     [(x_max, x_0 - space_x * i),
                                     (x_min, x_0 - space_x * i)], 1)
                self.render.draw_line(background, (220,220,220),
                                     [(x_max, x_0 + space_x * i),
                                     (x_min, x_0 + space_x * i)], 1)
                self.render.draw_line(background, (220,220,220),
                                     [(y_0 - space_y * i, y_max),
                                     (y_0 - space_y * i, y_min)], 1)
                self.render.draw_line(background, (220,220,220),
                                     [(y_0 + space_y * i, y_max),
                                     (y_0 + space_y * i, y_min)], 1)

        if draw_plot_lines:
            self.draw_plots(background, index, 1)

        # Draw center lines
        if draw_center:
            self.render.draw_line(background, (0,0,0),
                                 [(x_max, x_0),
                                 (x_min, x_0)], 3)
            self.render.draw_line(background, (0,0,0),
                                 [(y_0, y_max),
                                 (y_0, y_min)], 3)
        # Draw borders
        self.render.draw_line(background, (0,128,128),
                             [(x_min, y_max),
                             (x_max, y_max)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(x_min, y_min),
                             (x_max, y_min)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(x_max, y_max),
                             (x_max, y_min)], 3)
        self.render.draw_line(background, (0,128,128),
                             [(x_min, y_max),
                             (x_min, y_min)], 3)

        return background

    def plot_colorbar(self, total_range=73):
        for i in range(total_range):
            self.render.draw_rectangle(self.board,
                                       range_colorbar(total_range, total_range - i),
                                       [1250, 140 + 5 * i, 40, 5])

    def plot_text_titles(self):
        self.render.render_text(self.board, "Validator Lab",
                                (650, 25), fontSize=60)
        self.render.render_text(self.board, "Data visualization",
                                (320, 90), fontSize=30)
        self.render.render_text(self.board, "Space validator",
                                (820, 90), fontSize=30)
        self.render.render_text(self.board, "MSE interpolation plot",
                                (120, 570), fontSize=30)
        self.render.render_text(self.board, "MSE",
                                (450, 570), fontSize=25)
        self.render.render_text(self.board, "Gradient MSE",
                                (520, 570), fontSize=25)
        self.draw_point(self.board, 443, 577, colors['red'],
                        4, type='o')
        self.draw_point(self.board, 513, 577, colors['pink'],
                        4, type='o')
        self.render.render_text(self.board, "Validator MSE value",
                                (1190, 570), fontSize=25)
        self.draw_point(self.board, 1183, 577, (255, 165, 0),
                        4, type='o')
        self.render.render_text(self.board, "MSE value different validators",
                                (820, 570), fontSize=30)
        self.render.render_text(self.board, "MSE values",
                                (1240, 110), fontSize=25)
        self.render.render_text(self.board, "1.0",
                                (1300, 135), fontSize=25)
        self.render.render_text(self.board, "0.5",
                                (1300, 305), fontSize=25)
        self.render.render_text(self.board, "0.0",
                                (1300, 490), fontSize=25)
        self.render.render_text(self.board, "Image Visualization",
                                (1430, 95), fontSize=25)
        self.render.render_text(self.board, "Image Validator",
                                (1430, 315), fontSize=25)
        self.render.render_text(self.board, "i: Interpolation with validator",
                                (1430, 585), fontSize=20)
        self.render.render_text(self.board, "nums: Change validator",
                                (1430, 600), fontSize=20)
        self.render.render_text(self.board, "Label 0       ",
                                (1130, 90), fontSize=30)

        ## Coordinates
        self.messages_coord = [(575, 525), (1075, 525)]
        self.render.render_text(self.board, "x: None y: None       ",
                                self.messages_coord[0], fontSize=30)
        self.render.render_text(self.board, "x: None y: None       ",
                                self.messages_coord[1], fontSize=30)



    def showCoordinates(self, pos, precision=2):
        """
        Plot coordinates in the window
        :param ttt: instance window
        :param pos: (x, y)
        :param precision: Precision between coordinates (2 -> 0.001)
        :return:
        """
        x,y = pos
        self.x, self.y = x, y
        for i in range(len(self.messages_coord)):
            if self.x_min[i] > x or self.x_max[i] < x or \
            self.y_min[i] > y or self.y_max[i] < y:
                message = "x: None y: None       "
                self.on_click[i] = False
            else:
                self.on_click[i] = True
                x = x - self.x_0[i]  # Transform for graph
                x = x if x == 0 else round(x / (self.x_max[i] - self.x_0[i]),
                                           precision)
                y = -1 * (y - self.y_0[i])  # Transform for graph
                y = y if y == 0 else round(y / (self.y_max[i] - self.y_0[i]),
                                           precision)
                message = "x: {} y: {}       ".format(x,y)
            self.render.render_text(self.ttt, message,
                                    self.messages_coord[i],
                                    fontSize=30, background='white')
        self.render.refresh()


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

    def show(self):
        """
        Start the render and window loop
        :return:
        """
        pass

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

