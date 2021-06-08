from abc import ABC, abstractmethod

class render_interface(ABC):

    @staticmethod
    @abstractmethod
    def init_render(width, height, name):
        """
        Render initial window
        :param height: height
        :param name: name window
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def create_surface(ttt, color=(250, 250, 250),
                       transparent=False):
        """
        Initialize a window
        :param color: background color
        :param transparent: if is transparent
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def draw_line(surface, color, coordinates, r):
        """
        Create a line in a surface
        :param color: color of the line
        :param coordinates: coordinates line
        :param r: radius of the line
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def draw_rectangle(surface, color, coordinates):
        """
        Create a rectangle in a surface
        :param color: color of the line
        :param coordinates: coordinates line
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def draw_circle(surface, color, coordinates, r):
        """
        Create a circle in a surface
        :param color: color of the circle
        :param coordinates: coordinates circle
        :param r: radius of the circle
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def render_surface(ttt, surface, coordinates=(0, 0)):
        """
        Render a surface in a window
        :param surface: surface to render
        :param coordinates: coordinates where to render
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def render_image(ttt, path_image, coordinates):
        """
        Render a image in a window
        :param path_image: path of the image
        :param coordinates: coordinates where to render image
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def render_text(ttt, message, coordinates,
                    fontType=None, fontSize=23,
                    renderSize=60, color=(10, 10, 10),
                    background=None):
        """
        Render a text in a window
        :param message:
        :param coordinates:
        :param fontType:
        :param fontSize:
        :param renderSize:
        :param color:
        :param background:
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def refresh():
        """
        Refresh window
        :return:
        """
        pass
