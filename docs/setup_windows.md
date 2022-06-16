ハードウェア・ソフトウェア要件
=============

### Windows

|項目|要件|
|---|---|
|TensorFlow|TensorFlow 2.9.1|
|OS|Windows 7以上、Ubuntu 16.04以上、macOS 10.12.6以上|
|GPU|CUDAをサポートするNVIDIA製のGPUでかつ、<br>そのGPUがCompute Capability 3.5以上をサポートしている必要がある。<br><br>NVIDIA CUDA GPUs<br>https://developer.nvidia.com/cuda-gpus
|NVIDIA GPU drivers|NVIDIA グラフィックスドライバー 450.80.02以上|
|NVIDIA CUDA Toolkit|NVIDIA CUDA Toolkit 11.2|
|NVIDIA cuDNN|NVIDIA cuDNN 8.1.0|
|Python|Python 3.6～3.9|

セットアップ
=============

### Python仮想環境を構築する

```
$ python -m venv venv
```

### Pythonライブラリをインストールする

```
$ python -m pip install -r requirements_windows.txt
```

### プロジェクトのパスを環境変数に追加する

環境変数を設定し再起動する

```
PYTHONPATH = D:\Source\speedfighter-nft
```

### TensorFlow2をインストールする

TensorFlow2をインストールする  
https://www.tensorflow.org/install?hl=ja

セットアップが完了したら
=============

画像認識モデルの作成を進める