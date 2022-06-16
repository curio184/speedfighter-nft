使い方
=============

* ~/speedfighter/degit_recognition.py (画像認識モデルの作成スクリプト)
* ~/speedfighter/nft_issue.py (NFT発行機能のみを実行できるスクリプト)
* ~/speedfighter/speedfighter.py (アプリケーション本体)

画像認識モデルを作成する
=============

スピードガンに表示された数字を認識するため、  
画像認識モデルを作成・訓練するためのスクリプトを実行する。

```
~/speedfighter/degit_recognition.py
```

手順ごとにスクリプトを実行し、画像認識モデルを作成する。  

```
1-1. カメラで対象を録画し、教師データを収集する(Raspberry Piで実行)  
1-2. 録画を再生し、教師データを確認する(PCで実行)  
1-3. 教師データを分類し、データセットを作成する(PCで実行)  
1-4. データセットでモデルを訓練し、画像のカテゴリを覚えさせる(PCで実行)  
1-5. TFモデルに画像を入力し、学習結果を確認する(PCで実行)  
1-6. TF形式のモデルをTF LITE形式のモデルに変換する(PCで実行)  
1-7. TF Liteモデルに画像を入力し、学習結果を確認する(Raspberry Piで実行)  
```

すべての手順を実行すると、画像認識モデルが手に入る。

* keras_cnn_7segment_digits_28_28_gray_model.h5 (TensorFlowモデル)
* keras_cnn_7segment_digits_28_28_gray_model.tflite (TensorFlow LITEモデル)
* keras_cnn_7segment_digits_28_28_gray_class_names.json (カテゴリのラベル)

#### 1-1. カメラで対象を録画し、教師データを収集する(Raspberry Piで実行)

撮影画像は「./assets/rawdata/」に保存される。  

#### 1-2. 録画を再生し、教師データを確認する(PCで実行)

スクリプトを実行するとカテゴリ選択画面が起動する。  
適切なカテゴリをキー入力し、Enterで確定します。  
結果は「./assets/dataset/{class_name}/*.png」に出力される。  

#### 1-3. 教師データを分類し、データセットを作成する(PCで実行)

分類済みの画像セットをモデルに与えて、モデルを訓練する。  
訓練結果は「./assets/***_model.h5」に出力される。  
訓練されたモデルは与えられた画像のカテゴリを0-100%の確率として推測する。  

NFTの発行、NFTデータの復旧
=============

ノードやアカウント情報を設定する  
```
~/speedfighter-nft/speedfighter/configs/symbol.ini
```

NFTの発行、NFTデータの復旧を試してみる  

```
~/speedfighter/nft_issue.py
```

アプリケーション本体の実行
=============

### Supervisorを設定する

```
$ sudo apt install supervisor
```

/etc/supervisor/conf.d/speedfighter.conf 

```
[program:speedfighter]
command=/home/pi/Source/speedfighter-nft/venv/bin/python speed_fighter.py
user=pi
autostart=true
autorestart=false
stopsignal=INT
environment=PYTHONPATH=/home/pi/Source/speedfighter-nft
directory=/home/pi/Source/speedfighter-nft/speedfighter
chown=pi:pi
chmod=0777
```

```
$ sudo supervisorctl update
$ sudo supervisorctl start speedfighter
```

※supervisorから起動した場合、音がでない問題があり調査中。

### INIを設定し、実行する

```
~/speedfighter-nft/speedfighter/configs/speedfighter.ini  
~/speedfighter-nft/speedfighter/configs/symbol.ini  
~/speedfighter-nft/speedfighter/configs/twitter.ini
```
