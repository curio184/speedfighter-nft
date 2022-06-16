from typing import List

from cv2 import Mat
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.image_editor import ImageEditor


class SevenSegSplitter(AppBase):
    """
    7-Seg画像を数字ごとに分割するクラス
    """

    def __init__(self):
        super().__init__()

        # パラメータは手動で調整する
        self._offset = (0, 0)
        self._rotate = 0
        self._box_origin = (120, 200)   # (left, top)
        self._box_height = 215
        self._box_width = 107
        self._box_space = 13

    def draw_bounding_box(self, mat: Mat, titles: List[str] = None) -> Mat:
        """
        7-Seg画像を数字ごとに分割し、境界線を描画する
        """

        _mat = ImageEditor.copy(mat)

        ImageEditor.draw_rectangle(
            _mat,
            (self._box_origin[0], self._box_origin[1]),
            (self._box_origin[0] + self._box_width, self._box_origin[1] + self._box_height)
        )
        ImageEditor.draw_rectangle(
            _mat,
            (self._box_origin[0] + self._box_width * 1 + self._box_space * 1, self._box_origin[1]),
            (self._box_origin[0] + self._box_width * 2 + self._box_space * 1, self._box_origin[1] + self._box_height)
        )
        ImageEditor.draw_rectangle(
            _mat,
            (self._box_origin[0] + self._box_width * 2 + self._box_space * 2, self._box_origin[1]),
            (self._box_origin[0] + self._box_width * 3 + self._box_space * 2, self._box_origin[1] + self._box_height)
        )

        if titles:

            title_offset = (0, -5)

            ImageEditor.draw_text(
                _mat,
                titles[0],
                (
                    self._box_origin[0] + title_offset[0],
                    self._box_origin[1] + title_offset[1]
                )
            )
            ImageEditor.draw_text(
                _mat,
                titles[1],
                (
                    self._box_origin[0] + self._box_width * 1 + self._box_space * 1 + title_offset[0],
                    self._box_origin[1] + title_offset[1]
                )
            )
            ImageEditor.draw_text(
                _mat,
                titles[2],
                (
                    self._box_origin[0] + self._box_width * 2 + self._box_space * 2 + title_offset[0],
                    self._box_origin[1] + title_offset[1]
                )
            )

        return _mat

    def split_into_digits(self, mat: Mat) -> List[Mat]:
        """
        7-Seg画像を数字ごとの画像に分割する
        """

        mat1 = mat[
            self._box_origin[1]:self._box_origin[1] + self._box_height,
            self._box_origin[0]:self._box_origin[0] + self._box_width
        ]

        mat2 = mat[
            self._box_origin[1]:self._box_origin[1] + self._box_height,
            self._box_origin[0] + self._box_width * 1 + self._box_space * 1:self._box_origin[0] + self._box_width * 2 + self._box_space * 1
        ]

        mat3 = mat[
            self._box_origin[1]:self._box_origin[1] + self._box_height,
            self._box_origin[0] + self._box_width * 2 + self._box_space * 2:self._box_origin[0] + self._box_width * 3 + self._box_space * 2
        ]

        return [mat1, mat2, mat3]
