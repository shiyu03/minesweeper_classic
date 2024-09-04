import numpy as np


class CoordTransformer:
    def __init__(self, r1, c1, r2, c2):
        """
        物理坐标：r、c，原点在左上角，行向下，列向右。
        虚拟坐标：x、y，原点在左下角，x轴向右，y轴向上
        """
        if c1 == c2:
            self.rotate = np.array([[1, 0], [0, 1]])
            self.translate = np.array([-(r1-1), -(c1-1)])  # from (r1-1,c1-1) to (0,0)
        elif r1 == r2:
            self.rotate = np.array([[0, 1], [-1, 0]])
            self.translate = np.array([-(r1+1), -(c1-1)])  # from (r1+1,c1-1) to (0,0)
            self.translate = self.rotate.dot(self.translate)

        self.rotate_inv = np.linalg.inv(self.rotate)

    def p2v(self, r, c):
        """
        物理坐标：r、c，原点在左上角，行向下，列向右
        return 虚拟坐标：x、y，原点在左下角，x轴向右，y轴向上
        """
        physical = np.array([r, c])
        virtual = self.rotate.dot(physical) + self.translate
        return tuple(map(int, virtual))

    def v2p(self, x, y):
        """
        虚拟坐标：x、y，原点在左下角，x轴向右，y轴向上
        return 物理坐标：r、c，原点在左上角，行向下，列向右
        """
        virtual = np.array([x, y])
        physical = np.dot(self.rotate_inv, virtual - self.translate)
        return tuple(map(int, physical))
