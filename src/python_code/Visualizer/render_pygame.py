import pygame
from Visualizer.render_interface import render_interface

class pygame_render(render_interface):

    @staticmethod
    def init_render(width, height, name):
        """
        Render initial window
        :param height: height
        :param name: name window
        :return:
        """
        pygame.font.init()
        ttt =pygame.display.set_mode((width, height))
        pygame.display.set_caption(name)
        return ttt

    @staticmethod
    def create_surface(ttt, color=(250, 250, 250),
                       transparent=False):
        """
        Initialize a window
        :param color: background color
        :param transparent: if is transparent
        :return:
        """
        if transparent:
            surface = pygame.Surface(ttt.get_size(),
                                     pygame.SRCALPHA)
            surface = surface.convert_alpha()
        else:
            surface = pygame.Surface(ttt.get_size())
            surface = surface.convert()
            surface.fill(color)
        return surface

    @staticmethod
    def draw_line(surface, color, coordinates, r):
        """
         Create a line in a surface
         :param color: color of the line
         :param coordinates: coordinates line
         :param r: radius of the line
         :return:
         """
        pygame.draw.line(surface, color,
                         coordinates[0], coordinates[1], r)

    @staticmethod
    def draw_rectangle(surface, color, coordinates):
        """
        Create a rectangle in a surface
        :param color: color of the line
        :param coordinates: coordinates line
        :return:
        """
        pygame.draw.rect(surface, color, coordinates)

    @staticmethod
    def draw_circle(surface, color, coordinates, r):
        """
        Create a circle in a surface
        :param color: color of the circle
        :param coordinates: coordinates circle
        :param r: radius of the circle
        :return:
        """
        pygame.draw.circle(surface, color, coordinates, r)

    @staticmethod
    def draw_ellipse(surface, color, coordinates, witdh):
        """
        Create a ellipse in a surface
        :param color: color of the ellipse
        :param coordinates: coordinates ellipse
        :param witdh: witdh of the ellipse
        :return:
        """
        pygame.draw.ellipse(surface, color, coordinates, witdh)

    @staticmethod
    def render_surface(ttt, surface, coordinates=(0, 0)):
        """
        Render a surface in a window
        :param surface: surface to render
        :param coordinates: coordinates where to render
        :return:
        """
        ttt.blit(surface, coordinates)

    @staticmethod
    def render_image(ttt, path_image, coordinates):
        """
        Render a image in a window
        :param path_image: path of the image
        :param coordinates: coordinates where to render image
        :return:
        """
        carImg = pygame.image.load(path_image)
        ttt.blit(carImg, coordinates)

    @staticmethod
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
        font = pygame.font.Font(fontType, fontSize)
        if background is not None:
            text = font.render(message, renderSize, color,
            pygame.Color(background))
        else:
            text = font.render(message, renderSize, color)
        ttt.blit (text, coordinates)

    @staticmethod
    def refresh():
        """
        Refresh window
        :return:
        """
        pygame.display.flip()
