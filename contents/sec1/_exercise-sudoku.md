---
jupytext:
  formats: ipynb,_/md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: sdsadvml
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(sec:exercise-sudoku)=
# 演習 1 - 画像入力式数独ソルバーの作成

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ご存知の通り、数独というのは 9x9 のマスの中に 1-9 の数字を一定のルールのもとで入れていくパズルである。このルールとはすなわち、

- 同じ行に同じ数字が存在しない
- 同じ列に同じ数字が存在しない
- 3x3 の 9 領域に同じ数字が存在しない

の 3 つである。数独の解き方にはいろいろなやり方があり、

- バックトラック法
- 整数計画問題として解く
- Exact Cover 問題として解く

等のやり方がある。9x9 の問題であれば、最も単純なバックトラック法でも十分高速に解くことができる(一般的なパズル本の最難問なら Python 実装でも 1 秒以下で解ける)。

以下では、数独の問題を写真で撮って、その画像の中から問題を抜き出し、自動的に問題を解くプログラムを作成してみよう。

+++

```{warning}
「数独」という呼び名はパズル製作会社のニコリによって商標登録されている呼び名であり、別の呼び名として「ナンバープレイス」あるいは「ナンプレ」と呼ばれることもある。本資料では、一般的に最も浸透していると思われる呼び名である「数独」を用いる。
```

```{code-cell} ipython3
:tags: [remove-input]

"""
下準備のコード
"""

import re
import warnings
from copy import deepcopy

import seaborn as sns

try:
    from myst_nb import glue
except ImportError:
    glue = lambda *args, **kwargs: None

try:
    from advml.sudoku import solver

    has_advml = True
except ImportError:
    has_advml = False
    pass

# 警告の非表示
warnings.simplefilter('ignore', FutureWarning)

# グラフの設定
rc = {'figure.dpi': 150}
sns.set_theme(style='white', palette='colorblind', rc=rc)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## 数独を解く

+++ {"editable": true, "slideshow": {"slide_type": ""}}

まずは数独の問題がテキストで与えられている場合について、解き方を見ていこう。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

今回は数独の問題が以下の形式のテキストとして与えられることとする。Python は複数行のテキストを以下のように定義することができる。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = """
-35-9--48
--9--8--3
-4-6-5--1
----74---
-2-----6-
---15----
8--9-2-7-
9--5--2--
61--4-53-
"""
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

これを扱いやすくするために NumPy の配列に直してみる。まずはテキストを行ごとの配列に変換する。Pythonの文字列型のメソッドである`split`は空白文字や改行文字ごとに文字列を区切ってくれるのでこれを用いる。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = problem.split()
print(problem)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

するとproblemの長さは9となる。各行を文字ごとに分割するには、同じようにリスト内包表記を用いると簡単にできる。以下に二重ループを用いたリスト内包表記の例を示す。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = [c for line in problem for c in line]
print(problem)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

すると今度はproblemが81個の要素を持つ一次元配列になる。現在、各数字は文字であり、数字ではないので、各文字を数字に置き換える。また、まだ埋まっていないセルを表す`-`は 0 に置き替える。

このような置き換えは次のようなハッシュ(Python の用語では dict)を使うと簡単に書ける。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
numbers = {'-': 0}
numbers.update({str(i): i for i in range(1, 10)})
print(numbers)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

このハッシュを使って文字を数字に置き換えるには、やはりリスト内包表記を用いる。その後、NumPy の配列に直しておく。

ここまで一次元配列で作業をしてきたが、9x9 の二次元配列の方が以後の作業がやりやすいので、配列の形も直しておく。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np

problem = np.array([numbers[c] for c in problem], dtype='uint8')
problem = problem.reshape((9, 9))
print(problem)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

とすると、先ほどまでの文字が数字に置き換わるはずだ。なお、ここまで繰り返しリスト内包表記も用いたが、これらをまとめて 1 行で書くこともできる。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = """
-35-9--48
--9--8--3
-4-6-5--1
----74---
-2-----6-
---15----
8--9-2-7-
9--5--2--
61--4-53-
"""
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = np.array(
    [numbers[c] for line in problem.split() for c in line],
    dtype='uint8',
).reshape((9, 9))
print(problem)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

::::{admonition} 改行文字
:class: note

改行文字はOSによってデフォルトで使われているものが異なり、Windowsは`\r\n`、Macは`\r`、Linuxは`\n`が用いられている。これらの改行文字は`\r`がcarriage return (CR)と呼ばれ、`\n`がline feed (LF)と呼ばれることから、Windows、Mac、Linuxで用いられる改行をそれぞれCRLF、CR、LF改行とも呼ぶ。

プログラミングにおいては、改行文字の違いが問題となることも多いので、LFに統一する、あるいはCRLFに統一する、など自分でルールを決めてエディタ(VSCodeなど)を設定しておくと良い。
::::

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### 使用済みの数字を調べる

+++ {"editable": true, "slideshow": {"slide_type": ""}}

問題が数字の配列として表せたところで、使用済みの数字を調べる方法を見ていく。今`i`行`j`列にある数字の候補にどの数字が残っているのかを調べたいとする。

NumPy を用いると、`i`行目にある数字は`problem[i, :]`という配列で、`j`列目にある数字は`problem[:, j]`という配列で表せる。また 3x3 のブロックについては、ブロックの右上の座標が`(3 * (i // 3), 3 * (j // 3))`と表せることに注目すれば、以下でブロック内の数字が得られる。

```python
k = 3 * (i // 3)
l = 3 * (j // 3)
blk_nums = problem[k : k + 3, l : l + 3].flatten()  # 一次元配列として取り出す
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

一例として 1 行 1 列の空マスに何が入るのかを調べて見る。行、列、ブロックに使われている数字は、それぞれ以下のようになっている。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
row_nums = problem[0, :]
print('    Row #1:', row_nums)
col_nums = problem[:, 0]
print('    Col #1:', col_nums)
blk_nums = problem[0:3, 0:3].flatten()
print('Block #1-1:', blk_nums)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

まだ使える数字は 1-9 の数字の中で、これらの配列に現れていない数字である。配列中に数字があるかどうかは`in`を使うと調べられるので、リスト内包表記を使えば、未使用の数字が簡単に調べられる。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
used_nums = np.concatenate([row_nums, col_nums, blk_nums])
unused_nums = np.array([i for i in range(10) if i not in used_nums])
print('Unused numbers:', unused_nums)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

このように、`(0, 0)`の空マスに入りうる候補の数字は`{1, 2, 7}`の 3 つであることが分かる。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### バックトラック法による解法

+++ {"editable": true, "slideshow": {"slide_type": ""}}

最初に、数独をコンピュータに解かせる最も単純な方法を考えてみよう。

上記の使用済みの数字を調べる方法を少し変更すれば、各行、各列、各ブロックで数字が正しく 1 度だけ使われているかどうかを調べることができそうだ。とすれば、全ての空きマスに適当に数字を入れて、その結果が正しいかどうかを順に試せば良さそうに思える。

ただ、これでは正しい解が見つかる確率が低すぎるため、計算に時間がかかりすぎる。各マスに入れられる数字の候補は 9 通りでマスが 81 マスあるので、最悪の場合には$9^{81} \approx 2.0 \times 10^{77}$個の候補について調べる必要がある。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

一方で、実際にはいくつかのマスにはすでに数字が入っており、上記のように何も考えずに全てのパターンを試すのは非常に効率が悪いことが分かる。そこで、可能な候補についてだけ、しらみ潰しに調べることを考える。これを実現するのがバックトラック法である。

これでも効率がとても良いとは言えないが、実際には最初の方の数字がある程度決まると、後の方の数字は急速に候補が少なくなるため、実用的にはそれなりに高速に動作する。少なくとも 9x9 の問題であれば、十分に速い(16x16 の問題になるとそうはいかない！)

+++ {"editable": true, "slideshow": {"slide_type": ""}}

バックトラック法は**深さ優先探索**の一種で、数独の例で言えば、**候補となる数字を仮に入れる操作をできる限り繰り返し、もし最後のマスまで数字が入れば成功、途中で数字が入れられなくなったら失敗**として違う候補を試す、というものである。

深さ優先探索の実装方法は**再帰関数を使う方法**と FILO (first in last out, 最初に入れたものが最後に取り出されるということ) **データ構造 (一般にはスタックと呼ぶ) を用いる方法**の 2 つがある。

どちらの方法にも一長一短があるが、今回はスタックを用いる前提で話を進める。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

バックトラック法による数独の解法の流れは次のようになる。

1. **初期化:** スタックに入力の問題を追加
2. **深さ優先探索:**
   1. スタックに積まれた問題の途中経過を取り出す
   2. マスの中で、まだ空マスになっている最初のマスの候補を割り出す
   3. その空マスに各候補を入れたものをスタックに追加
   4. この際、もしマスが全て埋まっていたら、成功と見なし処理を終了

+++ {"editable": true, "slideshow": {"slide_type": ""}}

この部分の実装は演習課題とするが、上手く問題が解ければ、以下のような解答が得られるはずである。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
import matplotlib.pyplot as plt
from IPython import display
from matplotlib.animation import ArtistAnimation

if has_advml:
    fig, ax = plt.subplots()
    frames = []
    solution = solver.backtrack(problem, ax=ax, frames=frames)

    ani = ArtistAnimation(fig, frames, interval=100, blit=True)
    html = display.HTML(ani.to_jshtml())
    display.display(html)
    plt.close()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
print('Solution is:\n' + str(solution))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
%%capture perf
%%timeit -n 1 -r 10
solver.backtrack(problem)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# | label: avg_time_bt_simple

avg_time = ' '.join(perf.stdout.split(' ')[0:2])
print(avg_time)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

上記は、素直にバックトラック法を実装した場合の実行時間であるが、一回の問題を解くのに平均[](#avg_time_bt_simple)となっており、十分高速であることが分かる

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### 処理の効率化

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ある程度素直にバックトラック法を実装した場合でも、簡単な数独の問題であれば数十ミリ秒で解くことができる。だが、次の問題はどうだろう？

参照: <https://www.j-cast.com/premium/2018/10/19341452.html?p=all>

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
problem = """
---------
3--2-9--7
94--1--82
-5-6-2-3-
--63-74--
---------
---------
-82-3-59-
1--9-4--6
"""
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
problem = np.array(
    [numbers[c] for line in problem.split() for c in line],
    dtype='uint8',
).reshape((9, 9))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
print('Solution (simple backtrack):\n' + str(solver.backtrack(problem)))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
%%capture perf
%%timeit -n 1 -r 10
solver.backtrack(problem)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
# | label: avg_time_bt_hard
avg_time = ' '.join(perf.stdout.split(' ')[0:2])
glue('avg_time_bt_hard', avg_time)
```

```{code-cell} ipython3
print('Solution (recursive backtrack):\n' + str(solver.recursive(problem.copy())))
```

```{code-cell} ipython3
%%capture perf
%%timeit -n 1 -r 10
solver.recursive(problem.copy())
```

```{code-cell} ipython3
# | label: avg_time_recursive
avg_time = ' '.join(perf.stdout.split(' ')[0:2])
print(avg_time)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

この問題だと、上記の通り[](#avg_time_bt_hard)ほどの時間がかかる。

バックトラック法を高速化する手法には、候補の数が少ないものから埋めていく、などの様々な方法が考えられるので、各自、プログラムを書いて試してみてほしい。参考までに再帰関数を用いた実装に要する計算時間は[](#avg_time_recursive)であったので多少の高速化には成功しているだろうか。

+++

また、各自の確認用に難易度の異なる4問を示しておく。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# レベル1
problem = """
-35-9--48
--9--8--3
-4-6-5--1
----74---
-2-----6-
---15----
8--9-2-7-
9--5--2--
61--4-53-
"""
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# レベル2
# https://si-coding.net/sudoku10.html
problem = """
------1-4
4---78---
-3------2
5--4-7-3-
2------5-
9---264--
-27------
--56----7
-6--13---
"""
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# レベル3
# https://www.j-cast.com/premium/2018/10/19341452.html?p=all
problem = """
---------
3--2-9--7
94--1--82
-5-6-2-3-
--63-74--
---------
---------
-82-3-59-
1--9-4--6
"""
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# 世界一難しい？
# https://www.conceptispuzzles.com/ja/index.aspx?uri=info/article/424
problem = """
8--------
--36-----
-7--9-2--
-5---7---
----457--
---1---3-
--1----68
--85---1-
-9----4--
"""
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## 演習: 実用的な数独ソルバーの作成

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ここまでの内容を利用して、写真から数独の問題を読み取り、答えを返すプログラムを作成してみよう。

本番では、以下のレベル1からレベル3の内容で10問ずつ画像を与える。レベル1が1点、レベル2が2点、レベル3が3点の合計60点を総得点とする。各レベルの画像は以下のような性質を持つ。

- **レベル 1:** 画像は写真ではなくデジタル画像で、問題は常に同じ位置にある。
- **レベル 2:** 画像は写真で、問題は概ね真上から撮影されている。
- **レベル 3:** 画像は写真で、問題は斜めから撮影されており、なおかつページが多少湾曲している可能性がある。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
import cv2
import matplotlib.gridspec as gridspec

level1 = cv2.imread('../../data/sudoku/level_1.jpg')[:, :, ::-1]
level2 = cv2.imread('../../data/sudoku/level_2.jpg')[:, :, ::-1]
level3 = cv2.imread('../../data/sudoku/level_3.jpg')[:, :, ::-1]

fig = plt.figure(figsize=(12, 4))

a1 = level1.shape[1] / level1.shape[0]
a2 = level2.shape[1] / level2.shape[0]
a3 = level3.shape[1] / level3.shape[0]

# create subplots
gs = gridspec.GridSpec(1, 3, width_ratios=[a1, a2, a3])
p1 = plt.subplot(gs[0])
p2 = plt.subplot(gs[1])
p3 = plt.subplot(gs[2])

p1.imshow(level1)
p1.axis('off')
p1.set_title('Level 1')

p2.imshow(level2)
p2.axis('off')
p2.set_title('Level 2')

p3.imshow(level3)
p3.axis('off')
p3.set_title('Level 3')

plt.tight_layout()
plt.show()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

**入力の形式**

入力としては、NumPy の配列として`(H, W, 3)`の大きさ(H は画像の高さ、W は画像の幅)のカラー画像。また、写真に写り込む問題には以下のような制約がある。

- 画像中には「数独の問題」が 1 問だけ写っている
- 問題以外に余計な物が写っている可能性はあるが、問題が 2 問以上写っていることはない
- 問題は入力された画像の向きとおおよそ揃っており、問題が横を向いていたり、上下が逆であることはない
- 問題中の数字は必ず何らかの印刷用フォントを使って書かれており、手書きであることはない
- 画像中に映り込んでいる問題は必ず解が 1 つに定まる

**出力の形式**

数独の答えを表わす`(9, 9)`の大きさを持つ NumPy の配列。

**プログラムの作成方法**

これを入力とするような関数 `solve(image)` を作成し、テストサーバーにプログラムをアップロードすると、自動で点数が計算される。ただし、全体の実行時間が 3 秒を超えると、自動的にプログラムが終了し、0 点であると見なされる。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

**プログラムの例 (スケルトン)**

```python
import cv2
import numpy as np
import numpy.typing as npt


def solve(image: npt.NDArray[np.uint8], level: int) -> npt.NDArray[np.int32]:
    """
    Inputs:
      image: NumPy array with (H, W, 3) shape. The color channels are in RGB order.
      level: The difficulty level of the problem
    Output:
      9x9 NumPy array with 32-bit signed integer
    """
    # 以下、数独の問題を解く処理
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.zeros((9, 9), dtype='int32')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

**使用可能なライブラリ**

今回の実行環境では、以下のライブラリが使用可能となっている (Pip を使ってインストールされる)。これ以外のライブラリは使用できないので注意すること。

- numpy
- scikit-learn
- matplotlib
- pillow
- pandas
- opencv-python
- joblib

+++ {"editable": true, "slideshow": {"slide_type": ""}}

**テスト方法**

まずは、各レベル 5 問ずつ練習データを提供するので、それを用いてテストを行うと良い。以下の URL から ZIP ファイルをダウンロードして、講義中に伝えるパスワードで展開して使うこと。

- [数独演習 - 練習用データ](https://github.com/tatsy-classes/sudoku-solver-template/raw/master/data/samples.zip)

本番の採点は GitHub Classroom を使って行う。講義中で課題用の Classroom の URL を指示するので、その URL をブラウザで開き、テンプレートレポジトリを自分の GitHub アカウントと紐付ける。以後、課題用のレポジトリにコードを push する度にテストが走る。

本講義では、実際の問題として与えている画像を公開することはしないが、どの程度の画像に対してプログラムが正しく動いているかは、Github Actions のログを見ることで確認できる。

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### 高度な問題を解くためのヒント

+++ {"editable": true, "slideshow": {"slide_type": ""}}

今回の演習では

- 問題の領域を画像から切り出す
- 画像から機械分類により数字を識別する
- 読み取った問題から数独の解法を得る

という3つのタスクを解く必要がある。

そのそれぞれについて、上手く問題を解くための考え方の一端を紹介する。

+++

#### 問題領域の切り出し

+++ {"editable": true, "slideshow": {"slide_type": ""}}

問題領域の切り出しについては、 [図形の検出](#sec:figure-detection) の項で学んだ内容を参考に、数独問題の枠を検出し、セルの中の数字を切り離した画像を 81 枚得るようにするのが良い。レベル 1 の場合には、そのまま正方形の領域を切り出せば問題の領域が手に入る。

レベル 2 については、数独の問題にパースがかかっているので[図形の検出](#sec:figure-detection)の項を参考に、パースをキャンセルして数独問題が正方形領域に合うように逆射影変換を行う必要がある。この際、レベル 2 の画像には問題以外にも背景領域に何らかが映っている場合があるので注意すること。

レベル 3 のように問題が湾曲している場合には、問題全体に対して射影変換を考えても、問題を平面化することが難しいだろう。しかし、数字が含まれるセルごとに見ていくと、その湾曲の量は少量であり、おおよそ射影変換によってセルが歪んでいると近似できる。そこで、各セルを表す領域の四つ角を上手く検出し、数字ごとに逆射影変換を書けてセルごとの画像を得るのが良いだろう。

+++

#### 機械分類による数字の認識

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ここまで、数字の認識には MNIST を訓練データとして用いてきた。しかし、手書き文字と数独の問題に用いられているフォントとは大きな乖離があるため、MNIST で訓練した分類器ではフォントで書かれた文字を十分に識別できないだろう。

そこで、フォントで書かれた数字により構成されるデータセットを自分で作ることを考えてみてほしい。手書き文字と異なり、フォントで書かれた数字にはそれほどバラエティはないと考えられるので、そこまで多くの訓練データを用意しなくても、それなりの識別結果が得られるだろう。

+++

より多くの訓練データをプログラム的に作成するには、Pillowの`ImageDraw`を用いる方法がある。例えば、背景が白の画像を用意して、以下のように数字を書き込むことができる。

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
from PIL import Image, ImageDraw, ImageFont

h, w = 64, 64

# 画像作成する文字
text = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# フォントの設定
font = ImageFont.truetype('./fonts/MPLUS1-Regular.ttf', 40)

# 各文字について画像を生成する
imgs = []
for t in text:
    img = Image.new('L', (w, h), 'white')
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, h // 2), t, fill='black', font=font, anchor='mm')
    imgs.append(np.array(img))
```

```{code-cell} ipython3
:tags: [remove-input]

fig = plt.figure(figsize=(12, 4))

for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set(xticks=[], yticks=[])
    ax.imshow(imgs[i], cmap='gray')

plt.tight_layout()
plt.show()
```

データセット作成に用いるフォントについては、各自のシステムにインストールされているフォントを使うか、あるいは[Google Fonts](https://fonts.google.com/?subset=japanese&noto.script=Hira)で適当なフォントをダウンロードして使用すると良いだろう。

+++

#### 数独の問題を解く

+++ {"editable": true, "slideshow": {"slide_type": ""}}

数独の問題を解くアルゴリズムは冒頭で説明したバックトラック法で概ね問題ない。それよりも問題は機械識別の結果、数字を誤って読み取っている可能性がある、ということだろう。例えば、数字を読み取って時点で、各行や各列に同じ数字が含まれている場合があるかもしれないし、実際に解いてみると答えがなかったり、答えとなり得るものが複数になる場合もあるかもしれない (今回、画像に含まれている数独の問題は必ず解が 1 つに定まるものである)。

そこで、より多くの問題を解くために、誤った読み取りがあった場合にも、正解を導けるような工夫を検討してみてほしい。最初に識別器を学習したら、混同行列を用いて、どの数字をどの数字と勘違いしやすいのかを予め確認しておくことが大切である。その結果を用いて、例えば、あるセルが 7 で上手く解けなかったら、そのセルの数字を 1 に置き換えて解いてみるなどすると、正しく答えが求まるかもしれない。

実際の数独の問題は 9×9=81 のセルのうち、3-4割程度で30個前後の数字が含まれることになる。仮に識別器の識別精度が 90%であるとして、そのうち3個は読み間違えることになるので、識別器の精度を上げることはもちろん重要であるが、それ以上に、誤りを含む問題を上手く解く工夫が必要である。

+++

## 発展: Exact Cover による高速な解法

+++

バックトラック法よりも遙かに高速な解法として、数独の問題を Exact Cover の問題として定式化する方法がある。

Exact Cover 問題とは、とある集合$\mathcal{S}$と、その部分集合$s_1, \ldots, s_n$がいくつか与えられた時に、以下の 2 つの条件を満たすように部分集合を選ぶ問題である。

**条件 1:** 選ばれた部分集合$s_{i_1}, \ldots, s_{i_M}$の和集合が集合$\mathcal{S}$と同じになる

- $\mathcal{S} = \bigcap_{m=1}^M s_{i_m}$

**条件 2:** 選ばれた部分集合$s_{i_1}, \ldots, s_{i_M}$から任意の 2 つを選び出したとき、それらの積集合が空

- $s_{i_n} \cup s_{i_m} = \phi, \quad \forall n \neq m$

+++

```{admonition} Exact Coverの例
:class: note

一例として$\mathcal{S} = \{1, 2, 3, 4, 5\}$を考える。このとき、

- $s_1 = \{1, 2\}$
- $s_2 = \{2, 3\}$
- $s_3 = \{4, 5\}$
- $s_4 = \{ 3 \}$
- $s_5 = \{1, 3, 5\}$

であるなら、$s_1, s_3, s_4$の組み合わせは$\mathcal{S}$のExact Coverである。
```

+++

Exact Cover にはいくつかの解法があり、Donald Knuth が提案した Algorithm X と呼ばれる解法が広く知られている。この解法には、通常、Dancing Links という、二次元配列を上下左右に双方向連結リストで表現したようなデータ形式を用いる。Dancing Links には、配列の行や列の消去が効率的に行えるが、やや実装の難易度が高い。

今回は Dancing Links の代わりに Pandas の`DataFrame`を用いて、Exact Cover を Algorithm X によって解いてみる。まずは、単純な例として、上記の「Exact Cover の例」について、Algorithm X での解法を見てみる。

このデータを表で表わすと以下のようになる。

```{code-cell} ipython3
import numpy as np
import pandas as pd
from IPython import display

# DataFrameを作成し、データを0で初期化
df = pd.DataFrame(index=['s_1', 's_2', 's_3', 's_4', 's_5'], columns=[1, 2, 3, 4, 5])
df.iloc[:] = 0

# 部分集合の情報を入力
df.loc['s_1', [1, 2]] = 1
df.loc['s_2', [2, 3]] = 1
df.loc['s_3', [4, 5]] = 1
df.loc['s_4', [3]] = 1
df.loc['s_5', [1, 3, 5]] = 1

# 表の表示
display.display(df)
```

この表では、各部分集合について、値を持っている部分に 1 が、持っていない部分に 0 が入っている。Exact Cover が解くべき問題を表を用いて言い換えれば、「表の行のうちのいくつかを取り出して表を作ったときに、各列に必ず 1 つだけ 1 が存在する」ようにすること、となる。

これを実現するために、Algorithm X では、各列を見たときに、最も 1 が少ない列に注目する。この表では、「4」の列が最も 1 の数が少ない列である。なお、この操作の時に最も 1 が少ない列がもつ 1 の数が 0 である (つまる列の要素が全て 0 の列が存在する)場合には、Exact Cover は不可能なので処理を終了する。

```{code-cell} ipython3
def custom_style(df):
    index = df.index
    columns = df.columns
    styles = []
    for i in range(len(index)):
        s = []
        for j in range(len(columns)):
            if columns[j] == 4:
                s.append('background-color: pink; font-weight: bold;')
            else:
                s.append(None)
        styles.append(s)

    return np.array(styles, dtype='object')


df.style.apply(custom_style, axis=None)
```

このとき、仮に複数の列が同じ数の 1 を持っていたとしても、その操作のうち任意の一つだけを選択すれば良く、**それらを順々に処理する必要はない**。なぜなら、仮にここでその列を選択しなかったとしても、以後の処理で Exact Cover を実現するために、**いずれ残りの列も選択される**ためだ。

今回は、1 の数が一つだけの列は 4 列目だけなので、これを選んで処理を進める。選ばれた列に対して、値が 1 となっている「行」を順々に見ていく。今回は**s_3**の行だけが 1 を持っているので、この行を選択する。

```{code-cell} ipython3
def custom_style(df):
    index = df.index
    columns = df.columns
    styles = []
    for i in range(len(index)):
        s = []
        for j in range(len(columns)):
            if index[i] == 's_3' and columns[j] == 4:
                s.append('background-color: plum; font-weight: bold;')
            elif index[i] == 's_3':
                s.append('background-color: powderblue; font-weight: bold;')
            elif columns[j] == 4:
                s.append('background-color: pink; font-weight: bold;')
            else:
                s.append(None)
        styles.append(s)

    return np.array(styles, dtype='object')


df.style.apply(custom_style, axis=None)
```

選択された行の要素のうち、1 を持っている「列」に次は注目する。今回の例では、4, 5 行目が 1 の値を持っている。これらの列で、1 の要素を持っている「行」を洗い出す。すると、**s_3**, **s_5**の行が 1 の要素を 4, 5 列目の少なくともいずれかに 1 の値を持っている。

今、**s_3**の行を選択したということは、それが満たす条件である**4**, **5**の列は不要であり、また同じ条件を満たす**s_5**は Exact Cover を達成する上で選ばれてはいけないから、これらの行、列を削除して良いことが分かる。

```{code-cell} ipython3
def custom_style(df):
    index = df.index
    columns = df.columns
    styles = []
    for i in range(len(index)):
        s = []
        for j in range(len(columns)):
            if index[i] in ['s_3', 's_5'] and columns[j] in [4, 5]:
                s.append('background-color: plum; font-weight: bold;')
            elif index[i] in ['s_3', 's_5']:
                s.append('background-color: powderblue; font-weight: bold;')
            elif columns[j] in [4, 5]:
                s.append('background-color: pink; font-weight: bold;')
            else:
                s.append(None)
        styles.append(s)

    return np.array(styles, dtype='object')


df.style.apply(custom_style, axis=None)
```

ここまでの処理をまとめると、以下の手順になる。

1. 各**列**が持つ 1 の数を数えて、最も少ない 1 を持つ列の中の 1 つ(**列 X**とする)を選ぶ
1. 列 X の要素の中で 1 を持つ行を 1 つ(**行 A**とする)を選ぶ
1. 行 A の中で 1 の要素を持つ列 (**列 x, 列 y, ...** とする)を選ぶ
1. 列 x, 列 y, ...の各列において、少なくとも一つの 1 を持つ行 (**行 a, 行 b, ...** とする)を選ぶ

選ばれた **列 x, 列 y, ...** ならびに **行 a, 行 b, ...** を元の表から削除すると、小さな表が得られる。上記の例を用いると、色がついていないセルだけが残り、以下のような表が得られる。

```{code-cell} ipython3
new_df = df.loc[['s_1', 's_2', 's_4'], [1, 2, 3]]
new_df
```

このようにして、小さな表が得られたら、上記の手順 1-4 をこの表に対しても繰り返す。この表は、1 列目が、最も少ない 1 を持つので、選ばれる **列 x, 列 y, ...** ならびに **行 a, 行 b, ...** に色をつけると以下のようになる。

```{code-cell} ipython3
def custom_style(df):
    index = df.index
    columns = df.columns
    styles = []
    for i in range(len(index)):
        s = []
        for j in range(len(columns)):
            if index[i] in ['s_1', 's_2'] and columns[j] in [1, 2]:
                s.append('background-color: plum; font-weight: bold;')
            elif index[i] in ['s_1', 's_2']:
                s.append('background-color: powderblue; font-weight: bold;')
            elif columns[j] in [1, 2]:
                s.append('background-color: pink; font-weight: bold;')
            else:
                s.append(None)
        styles.append(s)

    return np.array(styles, dtype='object')


new_df.style.apply(custom_style, axis=None)
```

ここで、再度選ばれた行、列を削除すると、次は、以下のような表が得られる。

```{code-cell} ipython3
new_df2 = new_df.loc[['s_4'], [3]]
new_df2
```

このように表に対して、再度上記の操作を繰り返すと、**表全体が選択されて、空の表になる**ことが分かる (このとき、表の値が全て 1 であるとは限らないが、少なくとも 1 行は全ての列が 1 となる行が存在する)。このような空の表が現れたら処理を終了し、これまでに選択した列のインデックスの列を返すと、Exact Cover となる部分集合の列が得られる。

ここまでの処理で、手順 2 で選ばれた行を列挙すると、最初に**s_3**が、次に**s_1**が、そして最後に**s_4**が選ばれた。この三つの部分集合は元の集合$\mathcal{S}$の Exact Cover となっており、Algorithm X により問題が正しく解けていることが確認できた。

なお、Pandas の`DataFrame`を用いた実装は以下のようになる。

```{code-cell} ipython3
# Knuth's Algorithm X
def dfs(df: pd.DataFrame):
    # 使い勝手のためにNumPy配列に変換
    m = df.to_numpy()
    if m.size == 0:
        return []

    # 1の数が最小の行を探す
    sums = np.sum(m, axis=0)
    sc = np.argmin(sums)
    if sums[sc] == 0:
        return None

    sel_rows = np.where(m[:, sc])[0]
    for sr in sel_rows:
        mask_cols = np.where(m[sr, :])[0]
        mask_rows = [np.where(m[:, c])[0] for c in mask_cols]
        mask_rows = np.unique(np.concatenate(mask_rows))

        new_df = df.drop(index=df.index[mask_rows]).drop(columns=df.columns[mask_cols])
        ret = dfs(new_df)
        if ret is not None:
            return ret + [df.index[sr]]

    return None
```

```{code-cell} ipython3
solution = dfs(df)
if solution is not None:
    print('The solution for exact cover is:', sorted(solution))
else:
    print('Solution not found!')
```

### 数独の Exact Cover による定式化

+++

では、数独の問題を Exact Cover の問題として定式化してみよう。これには、数独の問題を少し違った見方から捕らえる必要がある。

まず、各セルに何らかの値が入った場合に何が起こるのかを考えてみる。とある A 行 X 列のセル AX に 1 が入ることは、

- セル AX には 1-9 のうち 1 が入るという**操作**がなされた
- セル AX に数字が入るという**条件**が満たされた
- A 行目に 1 が入るという**条件**が満たされた
- X 列目に 1 が入るという**条件**が満たされた
- セル AX が属するブロックに 1 が入るという**条件**が満たされた

という操作とそれによって充足される条件を意味する。これを集合の表記で書くとすれば、

$$
s_{\text{Row-A,Col-X}} = \{ c_{\text{Fill,Row-A,Col-X}}, c_{\text{1,Row-A}}, c_{\text{1, Col-X}} c_{\text{1, Blk-N}} \}
$$

のようになり、$c_{\text{Fill,Row-A,Col-X}}$は A 行 X 列に数字が入ることを、$c_{\text{1,Row-A}}$は行 A に 1 が入ることを、$ c*{\text{1, Col-X}}$はX列に1が入ることを、$c*{\text{1, Blk-N}}$はブロック N に 1 が入ることをそれぞれ表わす。

+++

このように数独の問題を捕らえ直すと、セルに数字を入れるという操作は、とある条件の全体集合の部分集合になっていることが分かる。

この際、操作の数は数字を入れるセルが 9×9=81 個あり、そこに 1-9 の 9 種類の数字が入りうるので、**729 個の操作があり得る**ことが分かる。

また、満たすべき条件としては、9×9 個のセルに数字が入る (81 通り)、9 個の各行に 1-9 の数字が入る (9×9=81 通り)、9 個の各列に 1-9 の数字が入る (9x9=81 通り)、9 個のブロックに 1-9 の数字が入る(9×9=81 通り)の合計**324 通りの条件がある**ことが分かる。

さらに、各条件は 1 つの操作によってしか満たされない (例: とある列に二つの 2 が入ることはあり得ない)ので、数独の問題が、729 個ある操作からいくつか (実際には 81 個)を選んで、324 個の条件をちょうど 1 度ずつ満たす問題、すなわち、Exact Cover の問題であることが分かる。

+++

以上の議論より、数独を Exact Cover の問題として解くには 729×324 の大きさの表を作り、数字が既知のセルに対応する条件に 1 を入れる。この表に対して Algorithm X を適用すれば、必要な操作(=どのセルにどの数字を入れるか)を決定することができる。

+++

以下では、各セルに数字を入れる操作を**R1C1#1**のように表わすこととし、セルに数字が入るという条件を**R1C1**、行に数字が入るという条件を**R1#1**、列に数字が入るという条件を**C1#1**、ブロックに数字が入るという操作を**B1#1**のように表わすことにする。

このルールに従うと、Algorithm X で処理すべき Pandas の`DataFrame`は以下のように作成できる。

```{code-cell} ipython3
index = [f'R{i:d}C{j:d}#{n:d}' for i in range(1, 10) for j in range(1, 10) for n in range(1, 10)]
cond1 = [f'R{i:d}C{j:d}' for i in range(1, 10) for j in range(1, 10)]
cond2 = [f'R{i:d}#{n:d}' for i in range(1, 10) for n in range(1, 10)]
cond3 = [f'C{j:d}#{n:d}' for j in range(1, 10) for n in range(1, 10)]
cond4 = [f'B{b:d}#{n:d}' for b in range(1, 10) for n in range(1, 10)]
conds = cond1 + cond2 + cond3 + cond4

df = pd.DataFrame(index=index, columns=conds, dtype='bool')
df.iloc[:] = 0
display.display(df)
```

この表に対して、数独の問題が全て空マスの場合の条件を代入していく。

```{code-cell} ipython3
for i in range(9):
    for j in range(9):
        for n in range(1, 10):
            r = i + 1
            c = j + 1
            b = (i // 3) * 3 + (j // 3) + 1
            op = f'R{r:d}C{c:d}#{n:d}'
            cd1 = f'R{r:d}C{c:d}'
            cd2 = f'R{r:d}#{n:d}'
            cd3 = f'C{c:d}#{n:d}'
            cd4 = f'B{b:d}#{n:d}'
            df.loc[op, [cd1, cd2, cd3, cd4]] = 1

display.display(df)
```

続いては、問題に既に入っている数字を元に表の操作と条件を書き換える。一例として、1 行 2 列に 3 という数字が入っていたとする。この場合、1 行 2 列に 3 以外の数字が入る、という条件、即ち**R1C2#1**や**R1C2#8**などは起こりえないので、これらを表の行から除去して良い。

また、1 行 2 列に 3 が入るということは、他の操作により 1 行目に 3 が入ったり、2 列目に 3 が入ったりすることはない、ということを意味する。従って、同様の条件を満たす可能性のある操作、即ち**R1C5#3**や**R4C2#3**なども表から消去して良いことが分かる。

今回は、先ほど 1 秒以上時間のかかった問題を例に、ルールに従って表を変形してみる。

```{code-cell} ipython3
problem = """
---------
3--2-9--7
94--1--82
-5-6-2-3-
--63-74--
---------
---------
-82-3-59-
1--9-4--6
"""

problem = np.array(
    [numbers[c] for line in problem.split() for c in line],
    dtype='uint8',
).reshape((9, 9))

print(problem)
```

```{code-cell} ipython3
for i in range(9):
    for j in range(9):
        if problem[i, j] != 0:
            r = i + 1
            c = j + 1
            b = (i // 3) * 3 + (j // 3) + 1
            n = problem[i, j]
            del_op1 = [f'R{r:d}C{c:d}#{n_:d}' for n_ in range(1, 10) if n != n_]
            del_op2 = [f'R{r_:d}C{c:d}#{n:d}' for r_ in range(1, 10) if r != r_]
            del_op3 = [f'R{r:d}C{c_:d}#{n:d}' for c_ in range(1, 10) if c != c_]
            del_op4 = [
                f'R{r_:d}C{c_:d}#{n:d}'
                for r_ in range(i // 3 * 3 + 1, i // 3 * 3 + 4)
                for c_ in range(j // 3 * 3 + 1, j // 3 * 3 + 4)
                if r != r_ or c != c_
            ]

            del_op = list(set(del_op1 + del_op2 + del_op3 + del_op4))
            df.drop(index=[id for id in del_op if id in df.index], inplace=True)

display.display(df)
```

この表を元に Exact Cover 問題の解法を Dancing Links(と類似したデータ構造)を用いて実装すると、正しい答えが、バックトラック法よりも数段高速に求まる。以下、ソースコードは示さないが、実行結果と、その際の実行時間を示す。プログラミングに自信のある読者は、ぜひ前述の`DataFrame`を使った例を参考に、より高速な実装に取り組んでほしい。

```{code-cell} ipython3
:tags: [remove-cell]

ops = {op: set() for op in df.index}
cds = {cd: set() for cd in df.columns}


for op in ops.keys():
    cds_for_op = df.columns[df.loc[op, :] == 1]
    ops[op].update(cds_for_op)

for cd in cds.keys():
    ops_for_cd = df.index[df.loc[:, cd] == 1]
    cds[cd].update(ops_for_cd)


if has_advml:
    rows = deepcopy(ops)
    cols = deepcopy(cds)
    solution = solver.exact_cover(rows, cols)
```

```{code-cell} ipython3
:tags: [remove-cell]

if has_advml:
    answer = np.zeros((9, 9), dtype='int32')
    for s in solution:
        pat = re.compile(r'R([1-9]{1})C([1-9]{1})#([1-9]{1})')
        mat = pat.search(s)
        r = int(mat.group(1))
        c = int(mat.group(2))
        n = int(mat.group(3))
        answer[r - 1, c - 1] = n

    print(answer)
```

```{code-cell} ipython3
:tags: [remove-cell]

%%capture perf
%%timeit -n 1 -r 10

if has_advml:
    rows = deepcopy(ops)
    cols = deepcopy(cds)
    solver.exact_cover(rows, cols)
```

```{code-cell} ipython3
:tags: [remove-cell]

# | label: avg_time_ec

if has_advml:
    avg_time = ' '.join(perf.stdout.split(' ')[0:2])
    print(avg_time)
else:
    print('Not available')
```

Exact coverに基づく解法を用いると、先ほどまで1秒以上かかっていた問題が、**たったの[](#avg_time_ec)で解けている**ことが分かる。このように、数独の問題を Exact Cover の問題と捉えることが、通常のバックトラック法よりも遙かに高速に問題を解けることが確認できた。

+++

```{admonition} なぜ速く解けるのか？
:class: note

今紹介したExact Coverを解くAlgorithm Xでは、表の列の中で**最も1の数が少ない列**を選んでいる。この操作が何を意味するかを考えてみよう。

表の列に1が入っている行というのは、その列に対応する操作を行うことで、列が表わす条件が満たされることを意味する。従って「最も1の数が少ない列」というのは「その列が表わす条件を満たすために選択しうる【セルへの数字の代入操作】の候補数が最も少ない」ということを意味する。

Algorithm Xは、ある意味では深さ優先探索を用いたバックトラック法の一種と考えることができるが、このように候補の数が最も少ない物から順に処理することで、深さが下がるごとに急速に表のサイズが小さくなるため、処理の効率が良い。これがExact Coverによる定式化をAlgorithm Xで解くと高速に動作する理由である。
```
