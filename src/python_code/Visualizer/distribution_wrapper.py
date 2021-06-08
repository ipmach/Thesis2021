
class Distribution:

    def __init__(self, x, y, z,name_labels):
        """
        Wapper distribution
        :param x: coordinate x
        :param y: coordinate y
        :param z: value
        :param name_labels: string list of labels names
        """
        self.original = None
        self.x = x
        self.y = y
        self.z = z
        self.labels = name_labels
