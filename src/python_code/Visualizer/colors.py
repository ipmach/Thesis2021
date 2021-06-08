colors = {"red": (255, 0, 0),
          "pink": (255, 0, 255),
          "blue": (0, 0, 255),
          "green": (0, 255, 0),
          "light blue": (0, 142, 255),
          "purple": (155, 0, 255),
          "dark green": (2, 120, 10),
          "dark red": (100, 0, 0),
          "light purple": (100, 0, 250),
          "cyan": (100, 250, 250)}

def range_colorbar(x, i):
    i = i if i <= x else x
    return (int(255/x) * i, 0, 255 - int(255/x) * i)
