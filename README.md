# aae-disentanglement
Use Adversarial Autoencoder just for disentanglement, with no arbitrary prior.

## Run
Tested on Python 3.8.10 + PyTorch 1.9.0. CUDA is required.

```
python3 ./train.py --dataset-path ~/data --latent-split 8 4 --dis-channels 128 --beta 0.25 --epoch 100
```

## Output
Here are the result images with various random seeds.
Rows consist of 8 channels block of latent variable z.
Columns consist of 4 channels block of latent variable z.
Diagonal ones are equal to raw output of the autoencoder, which is Decoder(Encoder(x)).

![run 1](images/dump0.png)
![run 2](images/dump1.png)
![run 3](images/dump2.png)
![run 4](images/dump3.png)
![run 5](images/dump4.png)


<!-- 
敵対的学習によってDisentanglement制約を与える手法を思いついて実装してみた。
画像はMNISTのAutoencoderでZを8+4次元に分解したもの。文字の太さや傾きなどの情報が分離されている様子がわかる。

Encoderの出力Zの複数のサンプルから、つぎはぎのZ'を作る。
DiscriminatorにZとZ'の2値分類を学習させる。
EncoderをDiscriminatorと敵対的に学習させることで、Zの断片同士の相互情報量に関する制約を与えることができる。

枠組みとしてはAdversarial Autoencoderと同じ。ただしターゲットの分布を定めずに潜在変数内での相互情報量のみについて制約を与える。
-->
