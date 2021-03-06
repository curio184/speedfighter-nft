世界初！NFT対応スピードガンspeedfighter！
=============

# プロジェクトの概要

市販のスピードガンを改造し、球速をカメラで読み取り、記録をNFTで発行します。

下記の技術を利用しています。
* スピードガン用マウントベースの作成は3D Printer
* 球速の文字認識にはTensorFlow
* 音声の読み上げにはVOICEVOX
* NFTの発行にはSymbolブロックチェーン

使用したスピードガンはこちら  
Bushnell 101911 Velocity Speed Gun

セットアップ手順と使い方
=============

私自身が使うためのプロジェクトのため最低限のざっくりです。

1. カメラで学習画像を撮影し、
2. 数字別に画像を分類、
3. TensorFlowでモデルを訓練し、
4. Symbolウォレットの準備ができたらセットアップは完了です。

詳細は下記ドキュメントを参照  
./docs/setup_windows.md  
./docs/setup_raspberry_pi.md  
./docs/how_to_use.md
