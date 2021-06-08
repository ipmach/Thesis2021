class ConvertPlot():

    def __init__(self, max_value_x, max_value_y, min_value_x,
                 min_value_y, max_range):
        max_y_ = max_value_y - min_value_y
        max_x_ = max_value_x - min_value_x
        self.x2y_ = lambda x: -1 * x + max_value_x
        self.y2x_ = lambda y: -1 * y + max_value_y
        self.y_2x = lambda y_: max_value_x - y_
        self.x_2y = lambda x_: max_value_y - x_
        self.y_2yo = lambda y_: (y_/max_y_) * max_range
        self.x_2xo = lambda x_: (x_/max_x_) * max_range
        self.yo2y_ = lambda yo: (yo * max_y_) / max_range
        self.xo2x_ = lambda xo: (xo * max_x_) / max_range

    def originalToPixel(self, x_list):
        x_list = list(map(lambda x: self.xo2x_(x), x_list))
        x_list = list(map(lambda x: self.x_2y(x), x_list))
        return x_list