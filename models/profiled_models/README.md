### Time Profiling
- lucina、1GPUでの結果です。
- モデルの選定のためにプロファイルをしたものはここに結果を保存しておきましょう。
- neneはいいですが、空いてないことも多いので再現実験したいときにできないので他のがオススメです。
-
|          | Faster-SCNN |  BiSeNet  | FasterSeg |
|inferrence| 7.0 ms      | 10.8 ms   |   ?       | 
|CPU + mask| 7.5 ms      | 47.5 ms   |   ?       | 

- inferrenceの部分が重要？　CPUに送り返すのと色分けの計算は変えられるかもしれないし、吐き出すデータ自体が時間かかる形式なのかもしれない。
 
