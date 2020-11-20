# BiSeNetV1 & BiSeNetV2

## 準備

### データセットの準備
ディレクトリをこんな感じにします。
```
-datasets/
        |-cityscapes/
                |-gtFine/
                |-leftImg8bit/
        |-signate/
                |-seg_train_annotations/
                |-seg_train_images/
        |-finetune/
                |-train/
                        |-img/
                        |-lb/
                |-val/
                        |-img/
                        |-lb/
        |-pretrain/
                |-train/
                        |-img/
                        |-lb/
                |-val/
                        |-img/
                        |-lb/               
        |-test/
        |-city2sig.py
```
この時、cityscapesとsignateには加工してないデータを入れてください。argumentで引いたデータセットが作られます。
```
$BiseNet/datasets
python city2sig.py --finetune --pretrain_mix --pretrain
```
これは、cityscapes のデータをsignate 仕様に変えてくれます。データセットを変えるときは、rootだけ変えば良くなります。mean とstdも自動で変更します。マスクのラベルも変更してくれます。くそ時間かかるので、tmuxとかでやったほうがいいよ。

### 環境の準備
デフォでは5GBあれば大丈夫（ラボ内GPUならall ok）。
cu+92で動くよ（というか、pipして）
```
$ Bisenet/
python -m venv venv
$ Bisenet/
source venv/bin/activate
$ Bisenet/
pip install -r requirements.txt
$ Bisenet/
deactivate
```
tmux内でactivateしっぱなしがおすすめだよ。その場合はもちろんdeactivateしなくていいです。opencvのせいで10分くらいはかかると思ったがいいよ。
## 注意
- クラス数が変わるとロスのスケールもだいぶ変わるので、lrは気をつけたほうがよさそう。
- datasetの準備以外のスクリプトはBisenet/で動かしてね

## Pretrain
```
$ BiseNet/
CUDA_VISIBLE_DEVICES=4 python tools/train.py --dataset_root datasets/pretrain_mix/train --lr 5e-3 
```
## Finetune
--finetune_fromのところを変えてください。
```
$ BiseNet/
CUDA_VISIBLE_DEVICES=4 python tools/train.py --dataset_root datasets/finetune/train --dataset signate --lr 1e-2 --finetune_from ./logs/res_2020_mm_yy_hh_mm/model_best.pth
```


## Test
- bisenetv2_lightを使ってください。これはtrainの時に出していた、補助関数をすべてカットしたものです。

test.pyで学習済みモデルにはかしてみる。1024のとこを変えてくだしあ
```
$ BiseNet
CUDA_VISIBLE_DEVICES=0 python tools/test.py --save_folder res/images/1024 --root datasets/finetune/val/img --exp logs/res_2020_mm_dd_hh_mm/
```
### jsonの作り方

まず、validationのgtをjsonにしたものを作る。res/jsons/gt.jsonができる
```
$ BiseNet
python signate_metrics/make_submit.py --path_to_annotations datasets/finetune/val/lb --output_name res/jsons/gt
```
さっきmodelに吐かせたやつをjsonにする。解像度とかをjsonの名前にしとくといいんじゃないですかね
```
$ BiseNet
python signate_metrics/make_submit.py --path_to_annotations res/images/1024 --output_name res/jsons/1024
```
とりあえず、iou出したいときは、
```
python signate_metrics/IOU.py --path_to_ground_truth res/jsons/gt.json --path_to_prediction res/jsons/1024.json
```

### マジの提出するやつ
まずは推論しましょう。
```
$ BiseNet
CUDA_VISIBLE_DEVICES=0 python tools/test.py --save_folder res/images/submission --root datasets/test
```
jsonを作る。
```
$ BiseNet
python signate_metrics/make_submit.py --path_to_annotations res/images/submission --output_name res/jsons/submit
```

はあぁ、牛角行きてえ
 