# city2sig
- cityscapesのマスクとイメージをsignate datasetと同じように使えるようにしました。
- 色が違うように見えますが、グレイスケールでのクラスIDは一緒になってるはずです。（違ったら、めちゃ教えて）
- cityscapesのstd, meanをsignate_testのものと同じにできるような関数も作りました。

## 使い方
- ソースコード中に書いてあります。関数の形にまとめたので、適宜使ってください

- main関数の中をいじって
```
python city2sig.py
```
- を実行してください
