# city2sig
- cityscapesのマスクとイメージをsignate datasetと同じように使えるようにしました。
- 色が違うように見えますが、グレイスケールでのクラスIDは一緒になってるはずです。（違ったら、めちゃ教えて）

## 使い方
- ディレクトリをこんな感じにしてください
```
datasets-/
         |-citys/
                 |-gtFine/
                 |-leftImg8bit/
signate_dataset/
         |-city4sig_pretrain/
                 |-gt/
                 |-img/
                 |-city2sig.py
```
- なお/はディレクトリを表します。

```
python city2sig.py
```
- を実行してください