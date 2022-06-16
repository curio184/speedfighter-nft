import copy
import math
from typing import List, Tuple

import cv2
import numpy as np
from cv2 import LINE_AA, Mat
from numpy.typing import NDArray
from PIL import Image, ImageOps


class ImageEditor:
    """
    画像エディター
    """

    @staticmethod
    def show(mat: Mat, window_name: str = "ImageEditor", milliseconds: int = 0):
        """
        画像を表示する
        """
        cv2.imshow(window_name, mat)
        cv2.waitKey(milliseconds)
        cv2.destroyAllWindows()

    @staticmethod
    def load_as_mat(file_path: str, as_gray: bool = False) -> Mat:
        """
        OpenCV形式で画像を読み込む
        使用用途: cv2

        Parameters
        ----------
        as_gray : bool, optional
            True: グレースケール画像として読み込み
            False: 3チャンネルカラー画像として読み込み
        """
        return cv2.imread(file_path, 0 if as_gray else 1)

    @staticmethod
    def load_as_pil_image(file_path: str, as_gray: bool = False) -> Image.Image:
        """
        Pillow形式で画像を読み込む
        使用用途: keras, matplotlib, pyocr
        """
        pil_image = Image.open(file_path)
        if as_gray:
            pil_image = ImageOps.grayscale(pil_image)
        return pil_image

    @staticmethod
    def mat_to_pil_image(mat: Mat) -> Image.Image:
        """
        OpenCV形式をPillow形式に変換する
        """
        # カラーを入力した場合、カラーで出力する
        if len(mat.shape) == 3 and mat.shape[2] == 3:
            # BGRからRGBに変換する
            mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            return Image.fromarray(mat_rgb)
        # モノクロを入力した場合、モノクロで出力する
        elif len(mat.shape) == 3 and mat.shape[2] == 1:
            mat_gray = np.squeeze(mat, -1)
            return Image.fromarray(mat_gray)
        # モノクロを入力した場合、モノクロで出力する
        elif (len(mat.shape) == 2):
            return Image.fromarray(mat)
        else:
            raise Exception("Unknown format detected.")

    @staticmethod
    def pil_image_to_mat(pil_image: Image.Image) -> Mat:
        """
        Pillow形式をOpenCV形式に変換する
        """
        # カラーを入力した場合、カラーで出力する
        if pil_image.mode == "RGB":
            mat_rgb = np.asarray(pil_image)
            # RGBからBGRに変換する
            mat_bgr = cv2.cvtColor(mat_rgb, cv2.COLOR_RGB2BGR)
            return mat_bgr
        # モノクロを入力した場合、モノクロで出力する
        elif pil_image.mode == "L":
            mat_gray = copy.deepcopy(np.asarray(pil_image))
            return mat_gray
        else:
            raise Exception("Unknown format detected.")

    @staticmethod
    def save(mat: Mat, path: str):
        # ファイル書き出し
        cv2.imwrite(path, mat)

    @staticmethod
    def resize(mat: Mat, width: int, height: int) -> Mat:
        return cv2.resize(mat, (width, height))

    @staticmethod
    def copy(mat: Mat) -> Mat:
        return copy.deepcopy(mat)

    @staticmethod
    def draw_rectangle(
            mat: Mat,
            upper_left: Tuple[int, int],
            bottom_right: Tuple[int, int],
            color_bgr: Tuple[int, int, int] = (0, 0, 255),
            thickness: int = 2,
            line_type: int = LINE_AA
    ):
        """
        矩形を描画する
        """
        cv2.rectangle(mat, upper_left, bottom_right, color_bgr, thickness, line_type)

    @staticmethod
    def draw_text(
        mat: Mat,
        text: str,
        org: Tuple[int, int],
        font_face: int = cv2.FONT_HERSHEY_PLAIN,
        font_size: int = 2,
        color_bgr: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        line_type: int = LINE_AA
    ):
        """
        テキストを描画する (※日本語不可)

        Parameters
        ----------
        org : Tuple[int, int, int]
            描画する位置、テキスト文字列の左下基準
        """
        cv2.putText(mat, text, org, font_face, font_size, color_bgr, thickness, line_type)

    @staticmethod
    def rgb_to_bgr(mat: Mat) -> Mat:
        """
        RGBをBGRに変換する
        ※OpenCVはBGR、PillowはRGB
        """
        return cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    @staticmethod
    def bgr_to_rgb(mat: Mat) -> Mat:
        """
        BGRをRGBに変換する
        ※OpenCVはBGR、PillowはRGB
        """
        return cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    @staticmethod
    def bgr_to_gray(mat: Mat) -> Mat:
        """
        BGRをGRAYに変換する
        """
        return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def gray_to_bgr(mat: Mat) -> Mat:
        """
        GRAYをBGRに変換する
        """
        return cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def mat_to_3d_array(mat: Mat) -> NDArray:
        """
        OpenCV形式を3次元配列(CNNの入力層の形状)に変換する
        ※OpenCVではモノクロ画像は2次元配列として読み込まれるため3次元配列に拡張する。
        """
        # 3次元カラー(w,h,3)の場合、変換不要
        if len(mat.shape) == 3 and mat.shape[2] == 3:
            return copy.deepcopy(mat)
        # 3次元モノクロ(w,h,1)の場合、変換不要
        elif len(mat.shape) == 3 and mat.shape[2] == 1:
            return copy.deepcopy(mat)
        # 2次元モノクロ(w,h)の場合、3次元モノクロ(w,h,1)に変換する
        elif len(mat.shape) == 2:
            return copy.deepcopy(np.expand_dims(mat, 2))
        else:
            raise Exception("Unknown format detected.")

    @staticmethod
    def threeed_array_to_mat(threed_array: NDArray) -> Mat:
        """
        3次元配列(CNNの入力層の形状)をOpenCV形式に変換する
        """
        # 3次元カラー(w,h,3)の場合、変換不要
        if len(threed_array.shape) == 3 and threed_array.shape[2] == 3:
            return copy.deepcopy(threed_array)
        # 3次元モノクロ(w,h,1)の場合、2次元モノクロ(w,h)に変換する
        elif len(threed_array.shape) == 3 and threed_array.shape[2] == 1:
            return copy.deepcopy(np.squeeze(threed_array, -1))
        # 2次元モノクロ(w,h)の場合、変換不要
        elif len(threed_array.shape) == 2:
            return copy.deepcopy(threed_array)
        else:
            raise Exception("Unknown format detected.")

    @staticmethod
    def normalize(mat: Mat) -> NDArray[np.float32]:
        """
        画像を0.0-1.0の範囲で正規化する
        """
        return mat.astype("float32") / 255.0

    @staticmethod
    def combine(mats: List[Mat]) -> Mat:
        """
        画像を結合する
        """

        # 入力画像を配置する枠(行と列数)を計算する
        # こんな順で行列を作る
        # [ 1][ 2][ 5]
        # [ 3][ 4][ 6]
        # [ 7][ 8][ 9]
        num_cols = math.floor(math.sqrt(len(mats)))
        num_rows = num_cols
        while True:
            num_cells = num_cols * num_rows
            if num_cells >= len(mats):
                break
            else:
                num_cols = num_cols + 1

            num_cells = num_cols * num_rows
            if num_cells >= len(mats):
                break
            else:
                num_rows = num_rows + 1

        # 入力画像の高さと幅を取得する
        height, width = mats[0].shape[:2]

        # 結合画像の高さ、幅、チャンネル
        max_height = 1080
        max_width = 1920
        combined_image_height = max_height if height * num_rows > max_height else height * num_rows
        combined_image_width = max_width if width * num_cols > max_width else width * num_cols
        col_width = math.floor(combined_image_width / num_cols)
        row_height = math.floor(combined_image_height / num_rows)
        combined_image = np.zeros((combined_image_height, combined_image_width, 3), dtype=np.uint8)

        # 入力画像を結合する
        idx_mat = 0
        for idx_row in range(1, num_rows + 1):
            for idx_col in range(1, num_cols + 1):
                if idx_row * idx_col <= len(mats):
                    # リサイズ
                    part = cv2.resize(mats[idx_mat], (col_width, row_height))
                    # 白黒の場合、カラーに戻す
                    part = part if len(part.shape) == 3 else cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
                    combined_image[
                        row_height*(idx_row-1): row_height*idx_row,
                        col_width*(idx_col-1): col_width*idx_col
                    ] = part
                    idx_mat = idx_mat + 1

        return combined_image
