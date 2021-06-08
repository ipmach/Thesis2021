from Visualizer.validation_figure import validationFigure
from sklearn.preprocessing import MinMaxScaler
from Visualizer.colors import colors
from tqdm import tqdm
import numpy as np
import pygame
from Visualizer.colors import range_colorbar
from Visualizer.plot_conversions import ConvertPlot
import pandas as pd
import cv2
from Visualizer.metric_functions import Metrics


class Validation(validationFigure):

    def __init__(self, general_model, models, D, space=10,
                 figsize=(1700, 950), number_lines=20):
        super().__init__(figsize=figsize, number_lines=number_lines)
        self.metrics = [Metrics.normalize_mse, Metrics.mse,
                        Metrics.euclidian_similarity, Metrics.structural_similarity,
                        Metrics.cosine_similarity, Metrics.PSNR]
        self.metricsNames = ["NMSE", "MSE", "Euclidian",
                             "SSIM", "Cosine", "PSNR"]
        self.metric_index = 0
        self.D = D
        self.general_model = general_model
        self.models = models
        self.scaler_x = []
        self.scaler_y = []
        self.scaler_x.append(MinMaxScaler(feature_range=(self.x_min[0] + space,
                                                    self.x_max[0] - space)))
        self.scaler_x.append(MinMaxScaler(feature_range=(self.x_min[1] + space,
                                                         self.x_max[1] - space)))
        self.scaler_y.append(MinMaxScaler(feature_range=(self.y_min[0] + space,
                                                    self.y_max[0] - space)))
        self.scaler_y.append(MinMaxScaler(feature_range=(self.y_min[1] + space,
                                                         self.y_max[1] - space)))
        self.name_labels = D.labels
        self.apply_metric()


    def apply_metric(self):
        self.render.render_text(self.board,
                                "Metric: " + self.metricsNames[self.metric_index] + "             ",
                                (1050, 45), fontSize=30, background='white')
        self.metric_use = self.metrics[self.metric_index]
        self.validators_data_mse = []
        self.validators = []
        for i in tqdm(range(len(self.models))):
            self.validators.append(self.models[i].do_encoder(
                self.D.original[i]))
            self.validators_data_mse.append(
                self.metric_use(
                    self.D.original[i],
                    np.array(self.models[i].do_decoder(
                        self.validators[-1]))))

    def plot(self, x, y, color, r, type, index, z=None, mse=None):
        """
        Plot a distribution in the graph
        :param x: Points coordinate x
        :param y: Points coordinate y
        :param color: color of the points (no labels)
        :param r: radius of the points
        :param type: type of shape of the points
        :param z: colors each label point
        :return:
        """
        x = np.array(x)
        y = np.array(y)
        x = self.scaler_x[index].fit_transform(
            x.reshape(len(x), 1)).reshape(-1)
        y = self.scaler_y[index].fit_transform(
            y.reshape(len(y), 1)).reshape(-1)
        if z is not None:  # We use labels
            keys = list(colors.keys())
            z = list(map(lambda x: colors[keys[x]], z))
            for j, x_y in enumerate(tqdm(list(zip(x,y)))):
                x_, y_ = x_y
                self.draw_point(self.ttt, int(x_), int(y_),
                                z[j], r, type=type)
        else:  # use MSE value
            for j, x_y in enumerate(tqdm(list(zip(x,y)))):
                x_, y_ = x_y
                self.draw_point(self.ttt, int(x_), int(y_),
                                range_colorbar(1, mse[j])
                                , r, type=type)


    def create_labels(self):
        """
        Print labels for the distribution
        :return:
        """
        self.render.render_text(self.ttt, "Labels:",
                                (self.x_min[4] + 20, self.y_min[4] + 20)
                                , fontSize=30)
        labels = self.render.create_surface(self.ttt,
                                            transparent=True)
        keys = list(colors.keys())
        for j,i in enumerate(self.name_labels):
            x = self.x_min[4] + 20
            y = self.y_min[4] + (j + 1) * 25 + 35
            self.render.render_text(self.ttt, "        " + i,
                                    (x,y), fontSize=25)
            self.draw_point(labels, x + 10,y + 8, colors[keys[j]],
                            6, type='o')
        self.render.render_surface(self.ttt, labels)

    def get_plotbox(self, mse):
        plot_box = []
        outliars_list = []
        max_num = 0
        for i in range(len(mse)):
            a = pd.DataFrame(mse[i]).describe()
            Q3 = a[0]["75%"]
            Q2 = a[0]["50%"]
            Q1 = a[0]["25%"]
            IQR = Q3 - Q1
            outliars = []
            aux = []
            for x in mse[i]:
                if Q1 - 1.5 * IQR < x < Q3 + 1.5 * IQR:
                    aux.append(x)
                else:
                    outliars.append(x)
            max_ = np.max(aux)
            if max_ > max_num:
                max_num = max_
            if np.max(outliars) > max_num:
                max_num = np.max(outliars)
            plot_box.append([Q1, Q2, Q3, max_,
                             np.min(aux)])
            outliars_list.append(outliars.copy())
        return plot_box, outliars_list, max_num

    def plot_box(self, x, y, mse=None):
        max_value_y = (self.x_min[3] + 50, self.y_max[3] - 32)
        min_value_y = (self.x_max[3] - 30, self.y_max[3] - 32)
        max_value_x = (self.x_min[3] + 50, self.y_max[3] - 30)
        min_value_x = (self.x_min[3] + 50, self.y_min[3] + 30)
        # Create plot
        self.render.draw_line(self.ttt, (0, 0, 0),
                              [max_value_y, min_value_y], 5)
        self.render.draw_line(self.ttt, (0, 0, 0),
                              [max_value_x, min_value_x], 5)
        i = len(self.models)
        dist = max_value_y[0] - min_value_y[0]
        middle_points = (np.arange(i) + 1) * dist / (i + 1)
        borders = [middle_points[0] / 2 + (dist / (i + 1)) * j
                   for j in range(i + 1)]
        plot_box, outliars, max_value = self.get_plotbox(x)
        conv = ConvertPlot(max_value_x[1],
                           max_value_y[0],
                           min_value_x[1] ,
                           min_value_y[0], max_value)
        self.draw_plots(self.ttt, 3, max_value, i=i)
        Q1 = list(map(lambda x: x[0], plot_box))
        Q2 = list(map(lambda x: x[1], plot_box))
        Q3 = list(map(lambda x: x[2], plot_box))
        max_ = list(map(lambda x: x[3], plot_box))
        min_ = list(map(lambda x: x[4], plot_box))
        Q1 = conv.originalToPixel(Q1)
        Q2 = conv.originalToPixel(Q2)
        Q3 = conv.originalToPixel(Q3)
        max_ = conv.originalToPixel(max_)
        min_ = conv.originalToPixel(min_)
        if mse is not None:
            mse = conv.originalToPixel(mse)
            mse = list(map(lambda x: int(x), mse))
        for j, i in enumerate(middle_points):

            # Draw plotbox
            self.render.draw_line(self.ttt, (0, 0, 0),
                                  [(max_value_y[0] - i + 5,
                                    max_[j]),
                                   (max_value_y[0] - i + 5,
                                    min_[j])], 2)
            self.render.draw_line(self.ttt, (0,0,0),
                                  [(max_value_y[0] - borders[j] + 10,
                                    max_[j]),
                                   (max_value_y[0] - borders[j + 1],
                                    max_[j])], 2)
            self.render.draw_line(self.ttt, (0, 0, 0),
                                  [(max_value_y[0] - borders[j] + 10,
                                    min_[j]),
                                   (max_value_y[0] - borders[j + 1],
                                    min_[j])], 2)
            self.render.draw_rectangle(self.ttt,
                                       colors["cyan"],
                                       [max_value_y[0] - borders[j] + 10,
                                       Q1[j],
                                       borders[j] - borders[j+1] - 10,
                                        Q3[j] - Q1[j]])
            self.render.draw_line(self.ttt, colors["red"],
                                  [(max_value_y[0] - borders[j] + 10,
                                    Q2[j]),
                                   (max_value_y[0] - borders[j+1],
                                    Q2[j])], 2)
            outliars[j] = conv.originalToPixel(outliars[j])
            for z in outliars[j]:
                self.draw_point(self.ttt, max_value_y[0] - i + 5, z,
                                (0,0,0)
                                , 2, type='x')

            #Plot points
            if mse is not None:
                if mse[j] < min_value_x[1]:
                    mse[j] = min_value_x[1]
                self.draw_point(self.ttt, int(max_value_y[0] - i + 5), mse[j],
                                (255, 165, 0)
                                , 4, type='o')
        self.render.refresh()

    def plot_image_model(self, point, size, model,
                         model2, name_1='foo.png', name_2='foo2.png'):
        """
        Save image to be plot in the board
        :param point: point in the window
        :param size: size of the image
        :param name: path where to save it
        :return: return path where to save it
        """
        img_ = model.do_decoder(np.array([point]))
        img = model.postformat(img_)[0]
        img = model.postprocessing(img)
        img = cv2.resize(img, size)
        cv2.imwrite(name_1, img)
        point = model2.do_encoder(img_)
        img2 = model2.do_decoder(np.array([point]))
        mse = self.metric_use(img_, img2[0])
        img2 = model2.postformat(img2)[0]
        img2 = model2.postprocessing(img2)
        img2 = cv2.resize(img2, size)
        cv2.imwrite(name_2, img2)
        return name_1, name_2, point, mse, img_

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
        for i in [0,1]:
            if self.on_click[i] == True:
                self.draw_point(self.ttt, self.x, self.y,
                                color, r, type=type)
                x = np.array(self.x).reshape(1, 1)
                y = np.array(self.y).reshape(1, 1)
                x = self.scaler_x[i].inverse_transform(x)[0][0]
                y = self.scaler_y[i].inverse_transform(y)[0][0]

                if self.interpolation > 0 and i == 0:
                    self.coor_inter.append([self.x, self.y, x, y])
                    self.interpolation += 1

                if i == 0:
                    name, name2, point, mse, img_ = self.plot_image_model([x, y],
                                                               reshape_image,
                                                               self.general_model,
                                                               self.models[self.start_index])
                if i == 1:
                    name2, name, point, mse, img_ = self.plot_image_model([x, y],
                                                                          reshape_image,
                                                                          self.models[self.start_index],
                                                                          self.general_model)
                self.render.render_image(self.ttt, name,
                                        (self.x_max[1] + 200,
                                         self.y_min[1]))
                self.render.render_image(self.ttt, name2,
                                         (self.x_max[1] + 200,
                                          self.y_min[1] + 220))
                self.render.render_text(self.ttt, "Metric reconstruction " + str(np.round(mse,2)) + "     ",
                                        (self.x_max[1] + 200, self.y_min[1] + 420), fontSize=25,
                                         background='white')
                i_ = int(not bool(i))
                x = int(self.scaler_x[i_].transform([[point[0][0]]])[0][0])
                y = int(self.scaler_y[i_].transform([[point[0][1]]])[0][0])
                if i == 0:
                    self.draw_point(self.ttt, x, y,
                                    range_colorbar(1, mse), 6, type='o')
                    self.draw_point(self.ttt, x, y,
                                    color, r, type='x')
                else:
                    self.draw_point(self.ttt, x, y,
                                    color, r, type='o')
                self.render.draw_rectangle(self.ttt,
                                           (255, 255, 255),
                                           [self.x_min[3] + 2,
                                           self.y_min[3] + 2,
                                           self.x_max[3] - self.x_min[3] - 6,
                                           self.y_max[3] - self.y_min[3] - 6])
                mse = []
                for m in self.models:
                    point = m.do_encoder(img_)
                    img2 = m.do_decoder(np.array([point]))
                    mse.append(self.metric_use(img_, img2[0]))
                self.plot_box(self.validators_data_mse,
                          list(range(len(self.validators_data_mse))),
                           mse=mse)

    def do_interpolation(self, samples=12, color=(192,192,192)):
        """
        Realize the interpolation
        :param samples: Number of samples of the interpolation
        :param color: Color of interpolation points
        :return:
        """
        if self.interpolation  == 3:
            self.interpolation = 0
            self.render.draw_line(self.ttt, color,
                                 [(self.coor_inter[0][0],
                                  self.coor_inter[0][1]),
                                 (self.coor_inter[1][0],
                                  self.coor_inter[1][1])], 1)
            self.render.draw_rectangle(self.ttt,
                                       (255, 255, 255),
                                       [self.x_min[2] + 2,
                                        self.y_min[2] + 2,
                                        self.x_max[2] - self.x_min[2] - 6,
                                        self.y_max[2] - self.y_min[2] - 6])
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
                self.draw_point(self.ttt,
                                int(self.coor_inter[0][0] + i * jump_x_p),
                                int(self.coor_inter[0][1] + i * jump_y_p),
                                color, 4)
                z_input.append([self.coor_inter[0][2] + i * jump_x,
                                self.coor_inter[0][3] + i * jump_y])
            z_input.append([self.coor_inter[1][2], self.coor_inter[1][3]])
            z_input = np.array(z_input)
            output = self.general_model.do_decoder(z_input)
            output_v = self.models[self.start_index].do_encoder(output)
            output_v = self.models[self.start_index].do_decoder(output_v)
            mse = []
            for o in range(len(output)):
                mse.append(self.metric_use(np.array([output[o]]),
                                            np.array([output_v[o]]))[0])

            index = 2
            max_value_y = (self.x_min[index] + 50, self.y_max[index] - 32)
            min_value_y = (self.x_max[index] - 30, self.y_max[index] - 32)
            max_value_x = (self.x_min[index] + 50, self.y_max[index] - 30)
            min_value_x = (self.x_min[index] + 50, self.y_min[index] + 30)
            max_value = np.max(mse)
            i = len(mse)
            dev_mse = []
            for j in range(1, len(mse)):
                dev_mse.append(mse[j] - mse[j - 1])
            if np.max(dev_mse) > max_value:
                max_value = np.max(dev_mse)
            self.draw_plots(self.ttt, index, max_value, i=i)
            conv = ConvertPlot(max_value_x[1],
                               max_value_y[0],
                               min_value_x[1],
                               min_value_y[0], max_value)
            mse = conv.originalToPixel(mse)
            mse = list(map(lambda x: int(x), mse))
            dev_mse = conv.originalToPixel(dev_mse)
            dev_mse = list(map(lambda x: int(x), dev_mse))
            dist = max_value_y[0] - min_value_y[0]
            middle_points = (np.arange(i) + 1) * dist / (i + 1)
            for j in range(1, len(mse)):
                if j > 1:
                    self.render.draw_line(self.ttt, colors["green"],
                                          [(int(max_value_y[0] - middle_points[j - 1] + 5),
                                            dev_mse[j-2] + 698),
                                           (int(max_value_y[0] - middle_points[j] + 5),
                                            dev_mse[j-1] + 698)], 4)
                self.render.draw_line(self.ttt, colors["cyan"],
                                      [(int(max_value_y[0] - middle_points[j - 1] + 5),
                                        mse[j - 1] + 698),
                                       (int(max_value_y[0] - middle_points[j] + 5),
                                        mse[j] + 698)], 4)
            for j in range(len(mse)):
                if j > 0:
                    self.draw_point(self.ttt, int(max_value_y[0] - middle_points[j] + 5),
                                    dev_mse[j-1] + 698,
                                    colors["pink"], 4, type='o')
                self.draw_point(self.ttt, int(max_value_y[0] - middle_points[j] + 5),
                                mse[j] + 698,
                                colors["red"], 4, type='o')

    def update_plot_validator(self, num):
        self.render.render_text(self.ttt, "Label " + str(num) + "       ",
                                (1130, 90), fontSize=30, background='white')
        self.render.draw_rectangle(self.ttt,
                                   (255, 255, 255),
                                   [self.x_min[1] + 2,
                                    self.y_min[1] + 2,
                                    self.x_max[1] - self.x_min[1] - 6,
                                    self.y_max[1] - self.y_min[1] - 6])
        self.create_plot(self.ttt, self.y_max[1], self.x_max[1],
                         self.y_min[1], self.x_min[1], self.y_0[1],
                         self.x_0[1], number_lines=20,
                         draw_center=True,
                         draw_plot_lines=False,
                         draw_squares=True,
                         index=0)
        self.start_index = num
        self.plot(self.validators[self.start_index][:, 0],
                 self.validators[self.start_index][:, 1],
                 (0, 0, 139), 2, 'o', 1,
                 mse=self.validators_data_mse[self.start_index])

    def show(self):
        """
        Start the render and window loop
        :return:
        """
        self.start_index = 0
        self.showBoard(self.ttt, self.board)
        self.plot(self.D.x, self.D.y, (0, 0, 139), 2,
                  'o', 0, z=self.D.z)
        self.plot(self.validators[self.start_index][:,0],
                  self.validators[self.start_index][:,1],
                  (0, 0, 139), 2, 'o', 1,
                  mse=self.validators_data_mse[self.start_index])
        self.create_labels()
        self.plot_box(self.validators_data_mse,
                      list(range(len(self.validators_data_mse))))
        self.render.refresh()
        running = True  # keep running
        reset = False
        self.interpolation = 0  # State interpolation
        total_labels = np.arange(len(self.name_labels ))
        while running:
            self.showCoordinates(pygame.mouse.get_pos())
            self.do_interpolation()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.plot_click()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i:
                        self.interpolation = 1
                        self.coor_inter = []
                    elif event.key == pygame.K_a:
                        running = False
                        reset = True
                        self.metric_index -= 1
                        if self.metric_index < 0:
                            self.metric_index = len(self.metrics) -1
                    elif event.key == pygame.K_d:
                        running = False
                        reset = True
                        self.metric_index += 1
                        if self.metric_index == len(self.metrics):
                            self.metric_index = 0
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3,
                                       pygame.K_4, pygame.K_5, pygame.K_6,
                                       pygame.K_7, pygame.K_8, pygame.K_9,
                                       pygame.K_0]:
                        label = int(chr(event.key))
                        if label in total_labels:
                            self.update_plot_validator(label)
        if reset: # reset with different metric
            self.apply_metric()
            self.show()
