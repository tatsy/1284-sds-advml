---
jupytext:
  formats: ipynb,_/md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: sdsadvml-dBAaNFbd-py3.10
  language: python
  name: python3
---

(sec:deep-q-learning)=
# 深層強化学習

+++

前節{ref}`sec:q-learning`ではTD学習を対象としてSARSAとQ学習について紹介した。

しかし、前節で解説したこれらの手法は**Qテーブルを離散化しなければならない**という欠点を持つ。

CartPoleの例では、浮動小数のパラメータ4つを8段階に量子化したために、状態空間の数は4096個であり、出力の操作の種類は右に動くか、左に動くかの2つであった。従って、Qテーブルのサイズは4096×2となる。

しかし、このテーブルのサイズは、パラメータや出力の数が増えたり、パラメータをより細かく離散化したりすると、急激にテーブルのサイズが増え、学習に時間がかかるだけでなく、そもそも状態空間の広さから学習が難しくなる、という問題があった。

そんな時にDeepMindの研究者らのチームによって公開された論文が「Playing Atari with Deep Reinforcement Learning」({cite}`mnih2013playing`)である。

そもそもニューラルネットワークは入出力がともに多次元の複雑な関数を表す能力に優れており、この論文ではニューラルネットによって、価値行動関数 $Q(s, a)$ を表現させている。このようなニューラルネットを**Qネットワーク**と呼ぶ。

```{code-cell} ipython3
:tags: [remove-cell]

"""
Google Colabの準備
"""

IN_COLAB = True
try:
    import google.colab

    print("You are running the code in Google Colab.")
except ImportError:
    IN_COLAB = False
    print("You are running the code on the local computer.")

if IN_COLAB:
    # Gymnasiumのインストール
    !pip install "gymnasium[classic-control]"
    pass
```

```{code-cell} ipython3
:tags: [remove-cell]

import random
from collections import deque

import cv2
import numpy as np
import seaborn as sns
import IPython.display as display
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.animation import ArtistAnimation

try:
    from myst_nb import glue
except ImportError:
    glue = lambda *args, **kwargs: None

# パラメータ
n_episodes = 200
glue("n_episodes", n_episodes)

# 乱数のシードを固定
random.seed(31415)
np.random.seed(31415)

# グラフの設定
rc = {
    "figure.dpi": 150,
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "legend.frameon": True,
    "legend.borderpad": 0.5,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
}
sns.set_theme(style="whitegrid", palette="colorblind", rc=rc)
```

## 深層Q学習の実装

+++

実は、DeepMind社の深層Q学習の論文以前にも、ニューラルネットワークを用いて強化学習をしよう、という試み自体は存在していた。

それらの手法は、シミュレーション環境から状態パラメータを受け取り、それを学習用データセットとしてためておいて、価値行動関数を表すQネットワークを訓練するというもので、この考え方は深層Q学習にも共通している。

これに対し、深層Q学習の論文では、

1. 状態パラメータを受け取らず、画像を入力としてプレイを行う
2. リプレイ・バッファを利用して、確率的最急降下法によりネットワークを訓練する

という2点が新しく提案されている。

深層Q学習の論文では、Atariゲーム (ATARI社が過去に開発したビデオゲーム)を題材としており、これらは所謂普通のビデオゲームであるため、状態パラメータを受け取ることはできず、**画像だけから、どのようなプレイを行なうかを判断しなければならない**。この点で、状態パラメータを受け取ることができる前節のCartPole環境より難しいタスクである。

また、状態パラメータが入力の場合も、画像が入力の場合も、時系列的に連続したデータから、ニューラルネットワークの訓練に用いるミニバッチを構成すると、勾配に強いバイアスがかかり、学習が進みづらくなるという問題がある。本論文では、**リプレイ・バッファと呼ばれる、過去の状態を記録しておくメモリを用意**しておき、その中からランダムに状態をサンプリングすることで、確率的最急降下法を可能としている。

以下では、まず状態変数をネットワークに入力する実装を紹介した後、画像だけを入力としてプレイを行なうAIへと改変する。

+++

### 下準備

+++

深層Q学習のコア部分を紹介する前に、いくつか下準備を行なっておく。まずは、PyTorchをインポートして、単純なニューラルネットワークを定義しておく。

なお、Q学習において、状態価値関数は任意の実数を取って良いので、最終層の活性化関数は不要である。

```{code-cell} ipython3
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class Network(nn.Sequential):
    """
    シンプルなmulti-layer perceptron
    """

    def __init__(self, n_inputs, n_outputs):
        super(Network, self).__init__(
            nn.Linear(n_inputs, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_outputs),
        )
```

深層Q学習においては、同じ構造を持つニューラルネットを2つ用意する。この理由については後述するが、訓練時にパラメータが更新されるネットワークが`q_net_online`であり、TD誤差の計算時に未来の状態のQ値を計算するのに用いられるのが`q_net_target`である。

また、Google Colab等のGPU環境でネットワークを訓練する場合を想定して、ネットワークのパラメータを指定したデバイスに送信し、その上で、オプティマイザを初期化する。

これはではオプティマイザとしてAdamを使用してきたが、深層強化学習においては、途中で学習データが更新されるため、Adamのような振動を防ぐ機構の入ったオプティマイザだと、訓練の後半で学習が進みづらくなる可能性がある。

このような問題を防ぐため、今回のコードでは単純なモメンタム付きの確率的最急降下法である`SGD`を用いることとする。

```{code-cell} ipython3
:tags: [remove-output]

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Your device is {device} ({torch.cuda.get_device_name(device)}).")
else:
    device = torch.device("cpu")
    print(f"Your device is {device}.")

# ネットワークの初期化
# CartPoleは状態変数の数が4で、出力パラメータ数が2
q_net_online = Network(4, 2)
q_net_target = Network(4, 2)
q_net_online.to(device)
q_net_target.to(device)

# オプティマイザの初期化
optim = torch.optim.SGD(q_net_online.parameters(), lr=1.0e-3, momentum=0.9)
```

この他、Gymnasiumの初期化やパラメータの設定は以下のように設定する。

```{code-cell} ipython3
import gymnasium as gym

# Q学習のパラメータ
gamma = 0.99

# 深層Q学習のパラメータ
batch_size = 32
steps_per_episode = 1000
memory_size = 10000

# ゲーム環境の作成
env = gym.make("CartPole-v1", render_mode="rgb_array")
```

```{code-cell} ipython3
:tags: [remove-cell]

glue("batch_size", batch_size)
glue("steps_per_episode", steps_per_episode)
glue("memory_size", memory_size)
```

今回の実験では、1プレイ (エピソード)ごとにサイズが{glue:}`batch_size`のミニバッチで{glue:}`steps_per_episode`ステップ分訓練を行う。

過去のゲームの状態を保存するリプレイメモリのサイズは最大{glue:}`memory_size`状態としておき、それ以後は古いものから順に捨てていくこととする。このようなリプレイ・バッファの実装は`collections.deque`を用いると容易である。

```python
from collections import deque

replay_buffer = deque(maxlen=memory_size)
```

+++

### エピソードの実行

+++

各エピソードでは、Gymnasiumから状態変数を受け取り、それをPyTorchのTensorに変更してQネットワークに渡す。状態変数の取得については前節で説明した通り。

```python
# 初期状態を取得する場合
s0, _ = env.reset()
# 新しい態を取得する場合 (a0は行動)
s1, reward, done, _, _ = env.step(a0)
```

CartPole環境においては状態変数の`s0`や`s1`は`float`値4つで表されているので、これを`torch.Tensor`型に変換する。なお、ニューラルネットワークに入力する際には型を`float32`型にした上で、デバイスを変更する必要がある。

```python
inputs = torch.Tensor(s0)  # Tensor型の作成
inputs = inputs.view(-1, 4)  # バッチ数に対応する次元を追加
inputs = inputs.float().to(device)  # 型とデバイスの変更
```

+++

オリジナルの深層Q学習におけるQネットワークの訓練では、$\varepsilon$-greedy法で行動選択を行う。

ランダムに行動する確率を`eps`で表した場合、次のようなコードで行動選択することになる。以下のコードにおいてネットワークの評価時はネットワークを`eval()`関数で評価モードに設定した上で`torch.no_grad()`スコープの中に入れて、自動微分による勾配計算をオフにしておく。

```python
# ε-greedy法
if np.random.rand() < eps:
    # ランダムな行動選択
    a0 = env.action_space.sample()
else:
    # ネットワークによるQ値の推定
    with torch.no_grad():
        q_net_online.eval()
        q_values = q_net_online(inputs)

    # NumPyの配列に変換して最大のQ値を持つ行動を選ぶ
    q_values = q_values.detach().squeeze().cpu().numpy()
    a0 = np.argmax(q_values)
```

+++

選択した行動でシミュレーションを更新したら、Qネットワークに用いる行動前の状態`s0`, 実際の行動`a0`, 行動後の状態`s1`, そのときに得られた報酬`reward`, ゲームの終了状態`done`をリプレイ・バッファに保存する。

```python
# 行動の選択
s1, reward, done, _, _ = env.step(a0)

# リプレイメモリに記録
replay_buffer.append((s0, a0, reward, s1, done))
```

+++

以上を実行すると`replay_buffer`に訓練用の状態データが蓄積される。

+++

### Qネットワークの訓練

+++

**Qネットワークの入出力**

前項{ref}`sec:q-learning`で解説した通り、Q学習は以下のTD誤差を最小化することを目指す。

$$
R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$ (eq:td-error)

+++

深層Q学習で $Q(s,a)$ をニューラルネットワークで表す際、入力を状態 $s$ と行動 $a$ の関数として表す代わりに状態 $s$ を入力として各行動に対する行動価値を返すようにモデル化している理由はここにある。

このTD誤差を計算において $\max_{a'} Q(s',a')$ という項を評価する際、Qネットワークが各行動に対する行動価値を表すベクトルを返すようにしておけば、SARSAのように次の行動 $s'$ を決めたり、全ての行動に対して $Q(s',a')$ を個別に評価する必要がなくなって効率が良いのである。

+++

**オンライン・ネットワークとターゲット・ネットワーク**

前述の通り、深層Q学習では同じ構造を持つ2つのネットワークを用意してQネットワークの訓練を行っていく。この際、パラメータを更新するネットワークを**オンライン・ネットワーク** (`q_net_online`)と呼び、パラメータを固定したネットワークを**ターゲット・ネットワーク** (`q_net_target`)と呼ぶ。

仮に単一のネットワークを用いて{eq}`eq:td-error`を定義すると、学習の過程において、{eq}`eq:td-error`中に現れる二つの $Q(s,a)$ に対して勾配が計算されてしまうため、学習が不安定になるという問題がある。実際、{eq}`eq:td-error`の二乗誤差を最小化しようとする場合、$Q(s, a)$は全ての行動に対して同じ行動価値を返すようにすれば最適解が得られてしまう。しかし、これではより良い行動をとる行動価値を学習したとは言えない。

そこで、深層Q学習では現在の行動に対する行動価値$Q(s, a)$を考える場合にはオンライン・ネットワークを用い、未来の行動に対する行動価値$Q(s',a')$を考える場合にはターゲット・ネットワークを用いる。

このようにして定義されるTD誤差は、いわば、現在ターゲット・ネットワークにより定義されている「未完成の」行動価値関数をベースとして、オンライン・ネットワークにより定義される行動価値関数を「より良い」ものにするための誤差であると言って良い。

ターゲット・ネットワークのパラメータは、オンライン・ネットワークのパラメータと一定間隔で同期してオンライン・ネットワークをさらに良いものへと更新していく。

以上の議論から、Qネットワークの訓練には、以下の誤差関数を用いる。

$$
\mathcal{L} = \mathbb{E}_{s, a, s'} \left[ \left( R(s, a) + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q_{\text{online}}(s, a) \right)^2 \right]
$$

ただし、ゲーム終了時においては、それ以後の状態を考慮する必要は無いため、以下のようなシンプルな誤差関数を用いれば良い。

$$
\mathcal{L} = \mathbb{E}_{s, a} \left[ \left( R(s, a) - Q_{\text{online}}(s, a) \right)^2 \right]
$$

+++

### リプレイ・バッファのデータセット化

+++

リプレイ・バッファのデータをPyTorchの枠組みでネットワークの訓練に用いるためには、カスタムの`Dataset`を用意しておくと良い。

```{code-cell} ipython3
class ReplayMemoryDataset(torch.utils.data.Dataset):
    """
    リプレイデータを1つずつ取り出すデータセット
    """

    def __init__(self, memory):
        self.memory = memory

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]
```

このデータセットを用いる場合、{ref}`sec:deep-learning`で紹介したように`DataLoader`型を直接初期化する方法がある。

```python
dataset = ReplayMemoryDataset(replay_buffer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

しかし、この方法だと、エピソードごとに学習するステップ数が変わってしまう (特に学習初期のリプレイデータが少ない時)ので、リプレイデータをランダムにサンプルしながら、事前に決めておいた{glue:}`steps_per_episode`回だけパラメータ更新を実行することにしよう。

このような`Dataset`からのランダムサンプルには`RandomSampler`クラスを用いる。以下のように`num_samples`に`batch_size`と`steps_per_episode`の積を入力しておくと、ミニバッチによる更新回数が`steps_per_episode`に一致するようになる。

```python
dataset = ReplayMemoryDataset(replay_buffer)
sampler = torch.utils.data.RandomSampler(dataset, num_samples=batch_size * steps_per_episode, replace=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
```

+++

### 実装例

+++

以上の議論を元にした深層Q学習の実装例を以下に示す。やや長いプログラムになるので初期状態では非表示としてある。

```{code-cell} ipython3
:tags: [hide-input, remove-output]

# ゲーム環境の作成
env = gym.make("CartPole-v1", render_mode="rgb_array")

# リプレイ・バッファの準備
replay_buffer = deque(maxlen=memory_size)

# 深層Q学習では, 線形にεを減少させる
e0 = 1.0
e1 = 0.005
epsilons = np.linspace(e0, e1, n_episodes)

# エピソードのループ
avg_steps = 0
pbar = tqdm(total=n_episodes * steps_per_episode)
for epi in range(n_episodes):
    # ゲーム環境のリセット
    s0, _ = env.reset()
    eps = epsilons[epi]

    # エピソード開始
    steps = 0
    while True:
        # ε-greedy法
        if np.random.rand() < eps:
            # ランダムに行動を選択
            a0 = env.action_space.sample()
        else:
            # Q-networkを使って行動を選択
            with torch.no_grad():
                q_net_online.eval()
                inputs = torch.Tensor(s0)
                inputs = inputs.unsqueeze(0).float().to(device)
                q_values = q_net_online(inputs)

            q_values = q_values.detach().squeeze().cpu().numpy()
            a0 = np.argmax(q_values)

        # 行動の選択
        s1, reward, done, _, _ = env.step(a0)

        # リプレイメモリに記録
        replay_buffer.append((s0, a0, reward, s1, done))

        # 次の状態に遷移
        s0 = s1
        steps += 1

        if done:
            break

    avg_steps = 0.9 * avg_steps + 0.1 * steps

    # データセットの用意
    memory_dataset = ReplayMemoryDataset(replay_buffer)
    memory_sampler = torch.utils.data.RandomSampler(
        replay_buffer,
        replacement=True,
        num_samples=batch_size * steps_per_episode,
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset,
        batch_size=batch_size,
        sampler=memory_sampler,
    )

    # 学習ループ
    q_net_online.train()
    for i, memory in enumerate(memory_loader):
        s0, a0, reward, s1, done = memory

        # 訓練データの型変換とデバイスへの転送
        s0 = s0.float().to(device)
        a0 = a0.long().to(device)
        reward = reward.float().to(device)
        s1 = s1.float().to(device)
        done = done.float().to(device)

        # Q値の計算
        q_values = q_net_online(s0)

        # 各行動の価値を取り出す
        q0 = torch.gather(q_values, 1, a0.unsqueeze(1)).squeeze(-1)

        # 次の状態に対するQ値の最大値を計算
        # この部分はターゲット・ネットワークを用いる
        with torch.no_grad():
            q_net_target.eval()
            q1 = q_net_target(s1)
            q_max = torch.max(q1, dim=1)[0]

        # 誤差関数の計算
        loss = F.mse_loss(q0, reward + gamma * q_max * (1.0 - done))

        # パラメータの更新
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 進捗状況の表示
        if i % 100 == 0:
            pbar.set_description(
                f"Episode {epi+1}/{n_episodes}, Steps: {avg_steps:.2f}, Loss: {loss.item():.3f}"
            )
        pbar.update()

    # Q-networkの更新
    if (epi + 1) % 5 == 0:
        q_net_target.load_state_dict(q_net_online.state_dict())
```

訓練済みのQネットワークを用いた実際のプレイの様子は次のようになる。

```{code-cell} ipython3
:tags: [remove-cell]

frames = []
obsrv, _ = env.reset()
while True:
    img = env.render()
    frames.append(img)

    # Q-networkを使ってQ値を計算
    inputs = torch.Tensor(obsrv)
    inputs = inputs.unsqueeze(0).float().to(device)
    with torch.no_grad():
        q_net_online.eval()
        q_values = q_net_online(inputs).detach().squeeze().cpu().numpy()

    # Q値が最大となる行動を選択
    a = np.argmax(q_values)

    obsrv, reward, done, _, _ = env.step(a)
    if done:
        break
```

```{code-cell} ipython3
:tags: [remove-input]

# アニメーションの描画
fig, ax = plt.subplots()
ax.set(xticks=[], yticks=[])

# 各フレームの描画
draw = []
for i, f in enumerate(frames):
    ims = plt.imshow(f)
    txt = plt.text(20, 30, f"frame #{i+1:d}")
    draw.append([ims, txt])
    fig.tight_layout()

# アニメーションの作成
ani = ArtistAnimation(fig, draw, interval=100, blit=True)
html = display.HTML(ani.to_jshtml())
display.display(html)

# Matplotlibのウィンドウを閉じる
plt.close()
```

以上のように行動価値関数$Q(s,a)$をニューラルネットによって表現することで、倒立振子の保持を大幅に長時間かすることに成功している。

+++

::::{admonition} 問
:class: question

上記の実装例において、ターゲット・ネットワーク (`q_net_target`)を用いずに誤差関数を定義した場合、どのように学習の様子や学習結果の挙動が変化するかを調査せよ。

::::

+++

## 画像を入力とした深層Q学習

+++

DeepMindのチームにより提案された深層Q学習の強みは、上記の実装のように状態変数を環境から読み取ることなく**プレイ画面のみから**必要な操作を出力できるという点にある。

画像を入力として扱う場合、上記の実装に加えていくつかの点に注意する必要がある。

+++

まず、当然ながらニューラルネットワークが画像を扱えるようにモデルを変更する必要がある。本稿では、以下のような単純な畳み込みニューラルネットワークを用いる。

```{code-cell} ipython3
class ConvNet(nn.Sequential):
    """
    シンプルな畳み込みニューラルネット
    """

    def __init__(self, in_channels, num_actions):
        super(ConvNet, self).__init__(
            # 畳み込み層 #1
            nn.Conv2d(in_channels, 128, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 畳み込み層 #2
            nn.Conv2d(128, 128, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 畳み込み層 #3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 畳み込み層 #4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 畳み込み層 #5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # 全結合層
            nn.Linear(128 * 9 * 9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )
```

ただし、ニューラルネットワークに入力する画像は**一時点のプレイ画面を表す画像1枚ではない**。CartPoleを例に取ると、状態変数には倒立振子の位置や傾きだけでなく、速度や各速度が与えられていた。

このような時間微分を要する状態変数を1枚の画像のみから予想することは難しい。そこで、画像を入力する場合には、とある時刻から過去に数フレームの情報をまとめてニューラルネットワークに入力する。

以下の実装では、プレイ画面をグレースケール化して1チャネルの画像とした後に4フレーム分の情報をニューラルネットワークに入力するため、**入力画像のチャネル数が入力フレーム数になっている**ことに注意してほしい。

```{code-cell} ipython3
# ネットワークの初期化
# 複数フレームの画像を入力する。出力パラメータ数が2
n_frames = 8
q_net_online = ConvNet(n_frames, 2)
q_net_target = ConvNet(n_frames, 2)
q_net_online.to(device)
q_net_target.to(device)

# オプティマイザの初期化
optim = torch.optim.SGD(q_net_online.parameters(), lr=1.0e-3, momentum=0.9)
```

また、各プレイ画面をオリジナルの解像度 (640×480ピクセル)のまま扱うと計算に時間がかかる上、消費メモリも大きい。

一方で、次の行動を決定する上では、画像の解像度が多少落ちていても問題ないと考えられるため、以下の実装では、画像を84×84ピクセルに縮小して入力する。

画像のグレースケール化、サイズの縮小、型の変更を含む前処理の関数は以下のようになるだろう。

```{code-cell} ipython3
def preprocess(img):
    """画像の前処理"""
    img = (img / 255.0).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img
```

最後に訓練に用いる誤差関数についてだが、画像を入力とする場合、タスクの難易度が上がるため、特に学習の初期において、出力される行動価値$Q(s, a)$の値が安定しないという問題が起こる。

この際、`F.mse_loss`で提供される平均二乗誤差を用いると、誤差の値が外れ値に強く反応してしまい、学習が進みづらくなる。

この問題を防ぐため、以下の実装では、平均二乗誤差の代わりに`F.smooth_l1_loss`で提供される**Huber損失**を用いる。Huber損失は0付近では平均二乗誤差のように動作し、0から離れた場所では平均絶対値誤差のように動作するため、外れ値の影響を受けづらいという特徴がある。

+++

以上の議論を踏まえた実装例を以下に示す。

今回の画像を用いる実装において、Qネットワークの訓練を十分に行うにはエピソード数を多めに設定する必要があり、また畳み込みニューラルネットワークの評価を含む各種計算に多くの時間を要する。

従って、本コードを実行する場合には、Google Colab等のGPU環境を用いることを推奨する。

```{code-cell} ipython3
:tags: [hide-input]

# ゲーム環境の作成
env = gym.make("CartPole-v1", render_mode="rgb_array")

# リプレイ・バッファの準備
replay_buffer = deque(maxlen=memory_size)

# 深層Q学習では, 線形にεを減少させる
e0 = 1.0
e1 = 0.005
epsilons = np.linspace(e0, e1, n_episodes)

# エピソードのループ
avg_steps = 0
pbar = tqdm(total=n_episodes * steps_per_episode)
for epi in range(n_episodes):
    # ゲーム環境のリセット
    env.reset()
    s0 = preprocess(env.render())
    eps = epsilons[epi]

    # エピソード開始
    recent_frames = deque(maxlen=n_frames + 1)
    for _ in range(n_frames + 1):
        recent_frames.append(s0)

    steps = 0
    while True:
        # ε-greedy法
        if np.random.rand() < eps:
            # ランダムに行動選択
            a0 = env.action_space.sample()
        else:
            # Q-networkを使って行動を選択
            with torch.no_grad():
                q_net_online.eval()
                state = np.stack(list(recent_frames)[1:], axis=2)
                inputs = torch.Tensor(state)
                inputs = inputs.unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
                q_values = q_net_online(inputs)

            q_values = q_values.detach().squeeze().cpu().numpy()
            a0 = np.argmax(q_values)

        # 行動の選択
        _, reward, done, _, _ = env.step(a0)
        s1 = preprocess(env.render())
        recent_frames.append(s1)

        # リプレイメモリに記録
        l = list(recent_frames)
        s0 = np.stack(l[:n_frames], axis=2)
        s1 = np.stack(l[1:], axis=2)
        replay_buffer.append((s0, a0, reward, s1, done))
        steps += 1

        if done:
            break

    avg_steps = 0.9 * avg_steps + 0.1 * steps

    # データセットの用意
    memory_dataset = ReplayMemoryDataset(replay_buffer)
    memory_sampler = torch.utils.data.RandomSampler(
        replay_buffer,
        replacement=True,
        num_samples=batch_size * steps_per_episode,
    )
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset,
        batch_size=batch_size,
        sampler=memory_sampler,
    )

    # 学習ループ
    q_net_online.train()
    for i, memory in enumerate(memory_loader):
        s0, a0, reward, s1, done = memory

        # 訓練データの型変換とデバイスへの転送
        s0 = s0.permute(0, 3, 1, 2).float().to(device)
        a0 = a0.long().to(device)
        reward = reward.float().to(device)
        s1 = s1.permute(0, 3, 1, 2).float().to(device)
        done = done.float().to(device)

        # Q値の計算
        q_values = q_net_online(s0)

        # 各行動の価値を取り出す
        q0 = torch.gather(q_values, 1, a0.unsqueeze(1)).squeeze(-1)

        # 次の状態に対するQ値の最大値を計算
        # この部分はターゲット・ネットワークを用いる
        with torch.no_grad():
            q_net_target.eval()
            q1 = q_net_target(s1)
            q_max = torch.max(q1, dim=1)[0]

        # 誤差関数の計算
        loss = F.smooth_l1_loss(q0, reward + gamma * q_max * (1.0 - done))

        # パラメータの更新
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 進捗状況の表示
        if i % 100 == 0:
            pbar.set_description(
                f"Episode {epi+1}/{n_episodes}, Steps: {avg_steps:.2f}, Loss: {loss.item():.3f}"
            )
        pbar.update()

    # Q-networkの更新
    if (epi + 1) % 5 == 0:
        q_net_target.load_state_dict(q_net_online.state_dict())
```

```{code-cell} ipython3
:tags: [remove-cell]

frames = []
env.reset()
s0 = preprocess(env.render())

recent_frames = deque(maxlen=n_frames)
for _ in range(n_frames):
    recent_frames.append(s0)

while True:
    img = env.render()
    frames.append(img)

    s0 = preprocess(img)
    recent_frames.append(s0)

    # Q-networkを使ってQ値を計算
    state = np.stack(list(recent_frames), axis=2)
    inputs = torch.Tensor(state)
    inputs = inputs.unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        q_net_online.eval()
        q_values = q_net_online(inputs).detach().squeeze().cpu().numpy()

    # Q値が最大となる行動を選択
    a = np.argmax(q_values)

    _, _, done, _, _ = env.step(a)
    if done:
        break
```

訓練後のQネットワークを用いた倒立振子の保持の様子は以下のようになる。

前述のものと同様、比較的長いプログラムになるので、初期状態では非表示としてある。

```{code-cell} ipython3
:tags: [remove-input]

# アニメーションの描画
fig, ax = plt.subplots()
ax.set(xticks=[], yticks=[])

# 各フレームの描画
draw = []
for i, f in enumerate(frames):
    ims = plt.imshow(f)
    txt = plt.text(20, 30, f"frame #{i+1:d}")
    draw.append([ims, txt])
    fig.tight_layout()

# アニメーションの作成
ani = ArtistAnimation(fig, draw, interval=100, blit=True)
html = display.HTML(ani.to_jshtml())
display.display(html)

# Matplotlibのウィンドウを閉じる
plt.close()
```

画像を入力とするタスクは直接状態変数を取得する場合と比べて難易度が大幅に上がるため、保持できる時間は短くなってしまっているが、それでも前項のQ学習と同程度の保持時間を実現することが出来ている。

この結果は、訓練を行うエピソード数を増やすことでさらに改善することが出来る。

+++

::::{admonition} 問
:class: question

本節で示した深層Q学習の性能を向上される方法は数多く提案されているが、その中で比較的簡単に実装できるものに**Dueling DQN** {cite}`wang2015dueling`と **優先度付き経験リプレイ** {cite}`schaul2015prioritized`がある。これらの技術について書籍やインターネット等で調査し、本節のコードに組み込むことで得られる性能を評価せよ。

::::

+++

::::{admonition} 問
:class: question

画像を入力とする深層強化学習を用いると、Gymnasiumのclassic-control以外の環境 (例えば[atari](https://ale.farama.org/environments/)など)でも、強化学習を実践することができる。実際に環境を変更することで、その効果を確かめよ。

ただし、Atariゲームの学習には非常に長い時間がかかるので、その点は注意が必要。Atariゲームの中では[Pong](https://ale.farama.org/environments/pong/)などが比較的易しいので、その辺りから試すと良い。

::::

+++

## 参考文献

```{bibliography}
:filter: docname in docnames
```

```{code-cell} ipython3

```
