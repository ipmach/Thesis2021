import pygame
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from Visualizer.colors import colors
import matplotlib.pyplot as plt
import cv2
from Visualizer.Figure_basic import Figure_basic
import math
from sklearn.metrics.pairwise import cosine_similarity


class Figure_latentSpace(Figure_basic):

    def plot_image_board(self, point, size, name='foo.png'):
        """
        Save image to be plot in the board
        :param point: point in the window
        :param size: size of the image
        :param name: path where to save it
        :return: return path where to save it
        """
        img = self.AE.do_decoder(np.array([point]))
        img = self.AE.postformat(img)[0]
        img = self.AE.postprocessing(img)
        img = cv2.resize(img, size)
        cv2.imwrite(name, img)
        return name

    def plot_click(self, color=(255, 165, 0),
                   reshape_image=(180,180), r=4, type='x'):
        """
        Click mouse  to add a point
        :param color: Color of the point add it
        :param reshape_image: size image generated
        :param r: radius of the point
        :param type: type of shape of the new point
        :return:
        """
        if self.on_click == True:
            self.draw_point(self.board2, self.x, self.y,
                            color, r, type=type)
            x = np.array(self.x).reshape(1, 1)
            y = np.array(self.y).reshape(1, 1)
            x = self.scaler_x.inverse_transform(x)[0][0]
            y = self.scaler_y.inverse_transform(y)[0][0]
            name = self.plot_image_board([x, y], size=reshape_image)
            self.render.render_image(self.ttt, name,
                                    (self.x_max + 50, self.y_min))
            # For interpolation
            if self.interpolation > 0:
                self.coor_inter.append([self.x, self.y, x, y])
                self.interpolation += 1
            # Insert image in plot
            if self.insert_image:
                name = self.plot_image_board([x, y], (50, 50))
                self.render.render_image(self.ttt, name,
                                         (self.x - 25, self.y - 25))

    def plot_plt(self):
        """
         Plot generate image in plt
        :return:
        """
        if self.img is not None:
            plt.imshow(self.img)
            plt.show()

    def create_labels(self):
        """
        Print labels for the distribution
        :return:
        """
        self.render.render_text(self.ttt, "Labels:",
                                (self.x_max + 50, self.y_0 ))
        labels = self.render.create_surface(self.ttt,
                                            transparent=True)
        keys = list(colors.keys())
        for j,i in enumerate(self.name_labels):
            x = self.x_max + 50
            y = self.y_0 + (j + 1) * 20
            self.render.render_text(self.ttt, "        " + i,
                                    (x,y))
            self.draw_point(labels, x + 10,y + 6, colors[keys[j]],
                            4, type='o')
        self.render.render_surface(self.ttt, labels)

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
        x = self.scaler_x.fit_transform(x.reshape(len(x), 1)).reshape(-1)
        y = self.scaler_y.fit_transform(y.reshape(len(y), 1)).reshape(-1)
        if z is not None:  # We use labels
            keys = list(colors.keys())
            z = list(map(lambda x: colors[keys[x]], z))
            for j, x_y in enumerate(tqdm(list(zip(x,y)))):
                x_, y_ = x_y
                self.draw_point(self.board, int(x_), int(y_),
                                z[j], r, type=type)
            self.labels=True
            self.name_labels = name_labels
        else:  # We dont use labels
            for x_, y_ in tqdm(list(zip(x,y))):
                self.draw_point(self.board, int(x_), int(y_),
                                color, r, type=type)

    def draw_tree(self, tree, draw_are=False, show_image=True):
        """
        Draw tree in the latent space
        :param tree: json file of tree
        :param draw_are: True to draw the area of each candidate
        :param show_image: True to show the image of each candidate
        :return:
        """
        points = []
        f = lambda x: [float(x[0]), float(x[1])]
        for k in tree:
           points.append(f(tree[k]['point']))
           x_ = self.scaler_x.transform([[points[-1][0]]])[0]
           y_ = self.scaler_y.transform([[points[-1][1]]])[0]
           self.draw_point(self.board, int(x_), int(y_),
                           (0,128,128), 7, type="o")
           if show_image: # Show images generated
               name = self.plot_image_board(points[-1], (50, 50))
               self.render.render_image(self.board, name,
                                        (x_ - 25, y_ - 25))
           if draw_are: # Show area each candidate
               try:
                   # Very subjective!!!
                   std = f(tree[k]['std'])
                   witdh_e = 100 * (x_ - std[0] * x_) / x_
                   high_e = 100 * (y_ - std[1] * y_) / y_
                   p_x = int(x_ - witdh_e /2)
                   p_y = int(y_ - high_e / 2)
                   self.render.draw_ellipse(self.board, (0,0,0),
                                            (p_x, p_y, witdh_e, high_e), 2)
               except:
                   print("Warning: enable to draw area")
           for c in tree[k]['children']:
               points2 = f(tree[str(c)]['point'])
               x_2 = self.scaler_x.transform([[points2[0]]])[0]
               y_2 = self.scaler_y.transform([[points2[1]]])[0]
               self.render.draw_line(self.board, (0,128,128),
                                    [[x_, y_], [x_2, y_2]], 4)

    def do_interpolation(self, samples=12, color=(192,192,192)):
        """
        Realize the interpolation
        :param samples: Number of samples of the interpolation
        :param color: Color of interpolation points
        :return:
        """
        if self.interpolation  == 3:
            self.interpolation = 0
            self.render.draw_line(self.board2, color,
                                 [(self.coor_inter[0][0],
                                  self.coor_inter[0][1]),
                                 (self.coor_inter[1][0],
                                  self.coor_inter[1][1])], 1)
            self.render.render_surface(self.ttt, self.board2, (0, 0))
            samples += 1
            jump_x = abs(self.coor_inter[1][2] - self.coor_inter[0][2])\
                    /(samples)
            jump_y = abs(self.coor_inter[0][3] - self.coor_inter[1][3])\
                    /(samples)
            jump_x_p = (self.coor_inter[1][0] - self.coor_inter[0][0])\
                    /(samples)
            jump_y_p = (self.coor_inter[1][1] - self.coor_inter[0][1])\
                    /(samples )
            z_input = []
            for i in range(samples):
                self.draw_point(self.board2,
                                int(self.coor_inter[0][0] + i * jump_x_p),
                                int(self.coor_inter[0][1] + i * jump_y_p),
                                color, 4)
                z_input.append([self.coor_inter[0][2] + i * jump_x,
                                self.coor_inter[0][3] + i * jump_y])
            self.render.render_surface(self.ttt, self.board2, (0, 0))
            z_input.append([self.coor_inter[1][2], self.coor_inter[1][3]])
            z_input = np.array(z_input)
            if self.insert_image:
                for j, z in enumerate(z_input):
                    x = int(self.coor_inter[0][0] + j * jump_x_p)
                    y = int(self.coor_inter[0][1] + j * jump_y_p)
                    name = self.plot_image_board(z, (50, 50))
                    self.render.render_image(self.ttt, name,
                                             (x - 25, y - 25))
            output = self.AE.do_decoder(z_input)
            output = self.AE.postprocessing(np.array(output))
            output_ = self.AE.postformat(np.array(output))
            fig = plt.figure()
            for i in range(len(z_input)):
                plt.subplot(4, math.ceil(len(z_input)/4), i+1)
                plt.imshow(output_[i])
            output = output.reshape((len(z_input),  28 * 28))
            # Calculate similarity
            similarity_first = cosine_similarity([output[0]], output)
            similarity_last = cosine_similarity([output[-1]], output)
            similarity_z_f = cosine_similarity([z_input[0]], z_input)
            similarity_z_l = cosine_similarity([z_input[-1]], z_input)
            # Make Interpolations plots
            plt.figure('similarity')
            plt.subplot(3,1,1)
            plt.title("Ouput similarity")
            plt.plot(similarity_first[0], label='similarity first')
            plt.plot(similarity_last[0], label='similarity last')
            plt.legend()
            plt.subplot(3,1,2)
            plt.title("Latent space similarity")
            plt.plot(similarity_z_f[0], label='similarity first')
            plt.plot(similarity_z_l[0], label='similarity last')
            plt.legend()
            plt.subplot(3,1,3)
            plt.title("Ratio between similarity")
            plt.plot(similarity_z_f[0]/similarity_first[0],
                     label='ratio between first')
            plt.plot(similarity_z_l[0]/similarity_last[0],
                     label='ratio between last')
            plt.legend()
            plt.show()

    def showOptions(self):
        """
        Show available options in the app
        :return:
        """
        message = "P: Plot generate image in plt "
        self.render.render_text(self.ttt, message,
                                (self.x_0 + 200, self.y_max + 20))
        message = "I: Do interpolation two points"
        self.render.render_text(self.ttt, message,
                                (self.x_0 + 200, self.y_max + 40))
        message = "O: Insert images in the latent space"
        self.render.render_text(self.ttt, message,
                                (self.x_0 + 200, self.y_max + 60))

    def show(self):
        """
        Start the render and window loop
        :return:
        """
        self.showBoard(self.ttt, self.board)
        self.showOptions()
        running = True  # keep running
        self.on_click = False  # CLick mouse is inside graph
        self.board2 = self.render.create_surface(self.ttt,
                                                 transparent=True)
        self.render.render_text(self.ttt,"Output",
                                (self.x_max + 50, self.y_min - 30),
                                fontSize=30)
        self.img = None  # Initialize plot plt
        self.interpolation = 0 # State interpolation
        self.insert_image = False # Insert image in teh screen
        while running:
          self.showCoordinates(self.ttt, pygame.mouse.get_pos())
          self.do_interpolation()
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.plot_click()
                self.render.render_surface(self.ttt, self.board2)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.plot_plt()
                elif event.key == pygame.K_i:
                    self.interpolation = 1
                    self.coor_inter = []
                elif event.key == pygame.K_o:
                    self.insert_image = not self.insert_image

    def __init__(self, AE, figsize=(900, 800), space=10,
                 space_right=200, number_lines=20):
        """
        Class with the possible operations in the latent space
        :param AE: AutoEncoder
        :param figsize: Size figure
        :param space: Space of the figure
        :param space_right: Space in the right size of the window
        :param number_lines: Number of lines drawing
        """
        super().__init__(figsize=figsize,
                         space_right=space_right)
        # Initialize scaler
        self.scaler_x = MinMaxScaler(feature_range=(self.x_min + space,
                                                    self.x_max - space))
        self.scaler_y = MinMaxScaler(feature_range=(self.y_min + space,
                                                    self.y_max - space))
        self.AE = AE

        # Initialize figure
        self.board = self.initBoard(self.ttt, number_lines=number_lines)
