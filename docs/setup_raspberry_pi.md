ハードウェア・ソフトウェア要件
=============

### Raspberry Pi 4

|項目|要件|
|---|---|
|TensorFlow|TensorFlow Lite 2.8|
|OS|Raspberry Pi OS 32bit|
|Python|Python 3.6～3.9|

セットアップ
=============

### Python仮想環境を構築する

```
$ python3 -m venv venv
```

### Pythonライブラリをインストールする

```
$ pip install --upgrade pip
$ pip install -r requirements_raspberry_pi.txt
```

### プロジェクトのパスを環境変数に追加する

```
$ sudo vim /etc/profile
export PYTHONPATH="/home/pi/Source/speedfighter-nft"
$ echo $PYTHONPATH
```

### pygame実行時にエラーが発生する場合

```
mixer module not available (ImportError: libSDL2_mixer-2.0.so.0: cannot open shared object file: No such file or directory)
NotImplementedError: mixer module not available (ImportError: libSDL2_mixer-2.0.so.0: cannot open shared object file: No such file or directory)
```

```
$ sudo apt install libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0
```

### ヘッドホン出力に切り替える   

```
$ sudo raspi-config
1. System Options > S2 Audio > 1. Headphones
```

セットアップが完了したら
=============

画像認識モデルの作成を進める