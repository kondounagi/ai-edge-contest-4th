# ai-edge-contest-4th
## トレーニングデータ
### 生画像
```
$ imgcat data/seg_train_images/train_0000.png
```
<img src="/pics/seg_train_images/train_0000.jpg">

### アノテーション
```
$ imgcat data/seg_train_annotations/train_0000.png
```
 <img src="/pics/seg_train_annotations/train_0000.png">

### メタデータ
```
$ cat data/seg_train_annotations/train_0000.json
{
    "attributes": {
        "route": "Tokyo1",
        "timeofday": "day"
    },
    "frameIndex": 4204
}
```

## 参考リンク
### signate
* [第4回AIエッジコンテスト](https://signate.jp/competitions/285)
* [第1回AIエッジコンテスト(セグメンテーション部門) 入賞者レポート](https://signate.jp/competitions/143/discussions/ai1-3)
* [第2回AIエッジコンテスト 入賞者レポート](https://signate.jp/competitions/191/summary)
  * 実際にFPGA上に実装した際のスコアまで評価対象としているのは，これまで第1~3回のうち第2回のみなので，第2回のレポートが実装のためには一番参考になりそう
* [第4回AIエッジコンテスト 実装チュートリアル](https://signate.jp/competitions/285#Tutorial)
  * とりあえずこれを参考に実装してみるのがよさそう

### Ultra96V2  
* [Ultra96V2向けVitis AI(2019.2)の組み立て方。](https://qiita.com/basaro_k/items/e71a7fcb1125cf8df7d2)
* [Ultra96 V2 に PYNQ の環境を構築して Jupyter Notebook へログインする](https://qiita.com/osamasao/items/cf0da1e53e633d4d8348)
* [PYNQ DLページ](http://www.pynq.io/board.html)

### 高位合成
* [NNgen](https://github.com/NNgen/nngen)
  * PythonでPyTorchのようにモデルを定義すると，そのモデルに適したVerilog HDLとIPコアを生成する
  * 理情の[高前田先生](https://sites.google.com/site/shinyaty/home-japanese)が中心となって制作
  * バックエンドは[Veriloggen](https://github.com/PyHDI/veriloggen)

### Semantic Segmentation Methods
* [Survey](https://www.sciencedirect.com/science/article/pii/S1568494618302813)
  * Real Timeの部分を見るとよいかも
* 精度とfpsはこんな感じ
  * <img src="/pics/fps-acc.png" width="320px">
* [awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
  * semantic-segmentationの歴代SOTAがまとめられている
  * 実装はない
* [Real-Time Semantic Segmentation | paper with code](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes)
  * Fastersegとかも良さそう
