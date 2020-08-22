# ai-edge-contest-4th

## 参考リンク
### signate
* [第4回AIエッジコンテスト](https://signate.jp/competitions/285)
* [第1回AIエッジコンテスト(セグメンテーション部門) 入賞者レポート](https://signate.jp/competitions/143/discussions/ai1-3)
* [第2回AIエッジコンテスト 入賞者レポート](https://signate.jp/competitions/191/summary)
  * 実際にFPGA上に実装した際のスコアまで評価対象としているのは，これまで第1~3回のうち第2回のみなので，第2回のレポートが実装のためには一番参考になりそう

### Ultra96V2  
* [Ultra96V2向けVitis AI(2019.2)の組み立て方。](https://qiita.com/basaro_k/items/e71a7fcb1125cf8df7d2)

### 高位合成
* [NNgen](https://github.com/NNgen/nngen)
  * PythonでPyTorchのようにモデルを定義すると，そのモデルに適したVerilog HDLとIPコアを生成する
  * 理情の[高前田先生](https://sites.google.com/site/shinyaty/home-japanese)が中心となって制作
  * バックエンドは[Veriloggen](https://github.com/PyHDI/veriloggen)
