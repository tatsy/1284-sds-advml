---
downloads:
- static: false
  title: Open in Google Colab
  url: https://colab.research.google.com/github/tatsy/1284-sds-advml/blob/main/contents/sec1/numpy.ipynb
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

(sec:numpy)=

# NumPyの基本

+++ {"editable": true, "slideshow": {"slide_type": ""}}

NumPy は一口に言えば、多次元の配列で表わされるデータ (例えば音声なら 1 次元の配列、画像なら 2 次元の配列)の処理と線形代数的な演算を簡易なコードで実行できるようにするライブラリといえる。まずは、Python の配列と NumPy の配列の違いについて見てみたい。

+++

Python で配列を作成するには`[ ]`で数字をコンマ区切りにすれば良く、

```{code-cell} ipython3
numbers = [1, 2, 3]
```

と書く。この時、a の各要素を 2 倍にしたいとすると、直感的には

```python
numbers * 2
```

とすれば良さそうだが、これは

```{code-cell} ipython3
:tags: [remove-input]

numbers * 2
```

のように配列が 2 回繰り返されたものとなってしまう。Python の配列でこれを実現するには**リスト内包表記**を使って

```{code-cell} ipython3
[x * 2 for x in numbers]
```

と書く必要がある。もちろん、これでも必要十分ではあるのだが、より直感的な記述で書ければ嬉しいだろう。そこで NumPy が登場する。

+++

NumPyは通常、`np`というエイリアスを設定してインポートするので、本資料もそれに従う。

NumPyを用いて配列を作る場合、Python の配列を引数にとる関数`np.array`を用いて、

```{code-cell} ipython3
import numpy as np

numbers = np.array([1, 2, 3])
```

と書けば良い。NumPy の配列であれば、下記のように要素を 2 倍にする計算を、より直感的に行うことができる。

```{code-cell} ipython3
numbers * 2
```

## NumPy の配列操作

+++

### 多次元配列を作る

+++

Python の配列で多次元配列を作るには、

```{code-cell} ipython3
numbers = [[1, 2], [3, 4]]
```

のように`[ ]`を二重にすれば良かった。同様の配列を NumPy で作るにはいくつかの方法があるが、最も単純には、上記の配列を`np.array`の引数に設定すれば良い。

```{code-cell} ipython3
numbers = [[1, 2], [3, 4]]
numbers_np = np.array(numbers)
```

また、全ての要素が 1 の 10x10 の二次元配列を作りたい場合、Python の配列では、リスト内包表記を用いて

```{code-cell} ipython3
zeros = [[0] * 10 for _ in range(10)]
```

のように書く必要があった。NumPy で同様の配列を作る場合、先ほど紹介した方法で`np.array`の引数に Python の多次元配列を代入する以外に`np.ndarray`という関数を使う方法がある。

`np.ndarray`は配列のサイズをタプルとして引数に取るので、10x10 の配列を作りたい場合には、引数に `(10, 10)`を指定すれば良い。

```{code-cell} ipython3
zeros = np.ndarray((10, 10))
```

### 特殊な配列の初期化

+++

NumPy では直接、Python の List を指定して配列を初期化する以外に、いくつかの便利な配列初期化方法が用意されている。ここでは、特によく用いるものをいくつか紹介する。

+++

#### 同じ値での初期化

```{code-cell} ipython3
print(np.zeros((5, 5)))  # 全て0で初期化
```

```{code-cell} ipython3
print(np.ones((5, 5)))  # 全て1で初期化
```

```{code-cell} ipython3
print(np.full((5, 5), 10.0))  # 全て同じ値 (10.0)で初期化
```

上記の関数を使う場合、`np.ndarray`の時と同様、関数の引数が次元配列の大きさを表す**タプル**になるので注意すること。

+++

#### 等差数列・等比数列による初期化

+++

等差数列によって NumPy の配列を初期化したい場合、Python に標準で用意されている`range(10)`などと同様の文法で`np.arange`を使うことができる。

```{code-cell} ipython3
print(np.arange(10))  # [0, 10)について1刻み
```

```{code-cell} ipython3
print(np.arange(5, 10))  # [5, 10)について1刻み
```

```{code-cell} ipython3
print(np.arange(0, 10, 2))  # [0, 10)について2刻み
```

```{code-cell} ipython3
print(np.arange(9, -1, -1))  # 9以下かつ-1より大きい整数を列挙
```

`np.linspace`を使うことで、上限と下限の間を固定の数で分割した等差数列を得ることもできる。このとき、上端を含むか、含まないかを`endpoint`引数により指定することもできる(初期値は`True`)。

```{code-cell} ipython3
print(np.linspace(0, 10, 5))  # 両端点を含み0.0と10.0の間を5分割
```

```{code-cell} ipython3
print(np.linspace(0, 10, 5, endpoint=False))  # 上端を含まず、0.0と10.0の間を5分割
```

同様にして、`np.geomspace`を用いると、上限と下限を指定した上で、固定の項数を持つ等比数列を得ることもできる。

```{code-cell} ipython3
print(geom := np.geomspace(1, 128, 8))  # 1から128までの項数が8の等比数列
```

```{code-cell} ipython3
print(np.log2(geom))  # 底が2の対数を取ると等差数列になっている
```

`np.geomspace`と類似した物として`np.logspace`もあり、`base`を基数とするべき乗の上付き添え字が等差数列になるような、等比数列を作成する。

```{code-cell} ipython3
print(logsp := np.logspace(0, 7, 8, base=2))  # 2^0から2^7までを8項で結ぶ等比数列
```

```{code-cell} ipython3
print(np.log2(logsp))  # log2を取ると、等差数列になっている
```

#### 二次元座標系の作成

+++

特に画像を取り扱う場合など、画素の次元を取得するために、二次元の離散的な座標系を作成したいことがよくある。例えば`ix`は`[[1, 2, ..., 10], ..., [1, 2, ..., 10]]`、`iy`は`[1, 1, ..., 1], [2, 2, ...., 2], ..., [10, 10, ..., 10]]`といった形だ。このような場合には`[1, 2, ..., 10]`という配列を作成した上で`np.meshgrid`を用いると良い。

```{code-cell} ipython3
ix = np.arange(0, 3)
iy = np.arange(0, 5)
ix, iy = np.meshgrid(ix, iy)  # ix, iyの順序に注意
print(f'ix=\n{ix}')
print(f'iy=\n{iy}')
```

また、NumPy には`np.mgrid`という変数が用意されていて、これはすでに全ての整数において`meshgrid`を用いたような二次元配列(のようなオブジェクト)となっていて、配列の範囲を指定することで`np.arange`を用いることなく二次元座標系を作成することができる。

```{code-cell} ipython3
iy, ix = np.mgrid[:5, :3]  # 軸の順序に注意
print(f'ix=\n{ix}')
print(f'iy=\n{iy}')
```

また、1 飛ばしで二次元座標系を作りたい場合、`np.arange`を用いれば、

```{code-cell} ipython3
ix = np.arange(0, 6, 2)
iy = np.arange(0, 10, 2)
ix, iy = np.meshgrid(ix, iy)
print(f'ix=\n{ix}')
print(f'iy=\n{iy}')
```

のように書けば良いが、`np.mgrid`を用いる場合には、スライスに与える添え字の書き方を工夫して以下のように書けば良い。

```{code-cell} ipython3
iy, ix = np.mgrid[0:10:2, 0:6:2]
print(f'ix=\n{ix}')
print(f'iy=\n{iy}')
```

上記の例に見られるような`[0:10:2]`といった添え字の書き方は**0 から始まり 2 刻みで 10 までの数字を添え字として配列要素を取り出す**という意味で様々な応用が可能なので、興味のある読者は発展的なスライスの使用法について調べて見てほしい。

+++

### 配列の情報

+++

上記の NumPy の配列を IPython 上で表示すると

```{code-cell} ipython3
numbers = np.array([1, 2, 3])
print(numbers)
```

のように表示されるが、この実態は`int64`型すなわち 64bit 符号付き整数となっている。これを調べるには，配列の`dtype`フィールドにアクセスすれば良く

```{code-cell} ipython3
numbers.dtype
```

のような出力が得られる。

また、配列の大きさは`shape`フィールドで、全要素数は`size`フィールドで、何次元の多次元配列なのかは`ndim`フィールドで調べることができる。

```{code-cell} ipython3
ones = np.ones((4, 5, 6))
print(f'Shape is {ones.shape}')
print(f'#elements is {ones.size}')
print(f'#dimensions is {ones.ndim}')
```

また、始めから配列要素の型を指定して、

```{code-cell} ipython3
numbers = np.array([1, 2, 3], dtype='float32')
```

のようにすることもできる。上記の例では、配列内の各要素が`float32`型、すなわち 32bit の単精度浮動小数で表わされる。

NumPy で使える配列の型には、この他にも`int8` / `uint8` (それぞれ 8bit 符号あり、符号なし整数)以下、`int16`、`int32`、`int64`が符号付き整数 (それぞれに符号なし整数である`uint..`が存在)の他、64bit 倍精度浮動小数として`double`、また複素数を表わす`complex64` (実部と虚部がそれぞれ 32bit 単精度浮動小数)や`complex128` (実部と虚部がそれぞれ 64bit 浮動小数)などがある。

最初に異なる型で宣言した配列を途中から別の型に変更したい場合には`astype`メソッドを使えば良い。

```{code-cell} ipython3
numbers = np.array([1, 2, 3])  # int32型
print(f'Original type is {numbers.dtype}')
numbers = numbers.astype('float32')  # 型をfloat32に変更
print(f'Updated type is {numbers.dtype}')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{admonition} オブジェクト指向型言語の用語
:class: note

Python はプログラミング言語の中ではオブジェクト指向型の言語に分類される (他には手続き型や関数型などがある)。オブジェクト指向型言語では、実世界のモノをプログラムにより表現するための仕組みとしてモノ(= オブジェクト)を「クラス」という概念で表わす。ここでは、詳しい明言を避けるが、Python のクラスとは以下のようなものだ。

```python
class Human(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def intro(self):
        print("Hi, my name is {:s}. I'm {:d} years old.".format(self.name, self.age))
```

このようなクラスにおいて、クラスが保持する変数 (= `self.name`や`self.age`)のことを**パラメータ**や**フィールド**と呼び、クラスが備える関数(= `intro`)のことを**メソッド**と呼ぶ。

また、上記のクラスを用いて、実際の`Human`を作ること、すなわち

```python
taro = Human('Taro', 22)
```

という処理を**インスタンス化**と呼ぶ。またインスタンス化の際に呼ばれるメソッドは`__init__`で定義されたもので、これを**コンストラクタ**と呼ぶ。

これらの用語は Python に限らず、オブジェクト指向型言語の基本用語なので覚えておくこと。
:::

+++

### 配列要素へのアクセス

+++

通常、Python で一次元配列、二次元配列の要素にアクセスするには

```python
print(arr1d[i])
print(arr2d[i][j])
```

のように要素を指定する。特に二次元配列に要素を指定するときには`[i][j]`のように`[ ]`を 2 つ使用して要素のインデックスを指定する。

一方で、NumPy を使う場合、一次元配列、二次元配列の要素へのアクセス方法は

```python
print(arr1d[i])
print(arr2d[i, j])
```

のようになり、特に二次元配列において、より簡素な表記で要素へのアクセスが可能となっている。

また、通常の Python と同様に`[:]`を指定することで、その次元の全ての要素を配列として取り出すことができる。例えば、以下のような使い方ができる。

```{code-cell} ipython3
arr2d = np.array([[1, 2], [3, 4]])
print('0th column:', arr2d[:, 0])
print('1st row:', arr2d[1, :])
```

配列要素の書き換えも、同様に可能で、特に`[:]`を指定した場合には、その部分配列の全ての要素を書き換えることができる。

```{code-cell} ipython3
# 例1: 単一要素の書き換え
arr2d = np.array([[1, 2], [3, 4]])
arr2d[0, 0] = 2
print(arr2d)
```

```{code-cell} ipython3
# 例2: 部分配列の書き換え
arr2d = np.array([[1, 2], [3, 4]])
arr2d[0, :] = 10
print(arr2d)
```

### 配列の変形

+++

NumPy では`reshape`を用いることで、配列の形を変更することもできる。例えば 2x2 の二次元配列を 4 要素の一次元配列に変形する場合、以下のように`reshape`を利用する。

```{code-cell} ipython3
arr2d = np.array([[1, 2], [3, 4]])
arr1d = arr2d.reshape(4)
print(arr1d)
```

また、変形後の大きさに`-1`を指定すると、その次元に限り、他の次元から自動的に要素数を計算してくれる。

```{code-cell} ipython3
arr2x3 = np.array([[1, 2, 3], [4, 5, 6]])
print('Before:\n', arr2x3)
arr3x2 = arr2x3.reshape((3, -1))
print('After:\n', arr3x2)
```

これを用いると、一次元配列への変換も、より容易に書くことができる。なお、一次元配列に変換する操作に限っては、`flatten`や`ravel`といった関数を使用する方法もある。

```{code-cell} ipython3
arr2d = np.array([[1, 2], [3, 4]])
print('reshape:', arr2d.reshape(-1))
print('flatten:', arr2d.flatten())
print('ravel:', arr2d.ravel())
```

:::{admonition} flatten と ravel の違い
:class: note

上記の通り、flatten と ravel はほとんど同じ機能を提供するが、得られる戻り値が、元の配列をコピーした「別の実態」なのか、それとも、単に元の配列の要素へのアクセスを一次元配列的にできるようにした「同じ実態」なのかが異なる。

結論から言えば、`flatten`は別の実態を、reshape と ravel は「可能な限り」同じ実態を返す。従って、要素の書き換えを行う場合には、注意が必要だ。以下のコードを実行して違いを確認してほしい。

```python
arr2d = np.array([[1, 2], [3, 4]])
arr1d_rs = arr2d.reshape((-1))
arr1d_fl = arr2d.flatten()
arr1d_rv = arr2d.ravel()
arr2d[0, 0] = 2
print('reshape:', arr1d_rs)
print('flatten:', arr1d_fl)
print('ravel:', arr1d_rv)
```

:::

+++

また、NumPy の配列は、次元を増やすことも可能で、例えば 2x2 の二次元配列を 2x1x2 の三次元配列にしたりできる。これには`reshape`を使うことができるほか、新しく追加する次元に対応する箇所に`None`あるには`np.newaxis`と書くことで次元を追加できる。

```{code-cell} ipython3
arr2d = np.array([[1, 2], [3, 4]])
print('reshape:', arr2d.reshape((2, -1, 2)).shape)
print('None:', arr2d[:, None].shape)
print('newaxis', arr2d[:, np.newaxis].shape)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## ベクトル・行列の演算

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### スカラに対する演算

+++ {"editable": true, "slideshow": {"slide_type": ""}}

ベクトルや行列に対して、スカラを四則演算すると、要素ごとに同じ計算が行われる。例えば、

```{code-cell} ipython3
a = np.array([1, 2])
print(a + 1)
```

の例では、a の各要素に 1 が足し算されていることが分かる。これは、他の四則演算に対しても同様である。

+++

### 要素ごとの四則演算

+++

同じ要素数を持つベクトル同士や行列同士を四則演算すると、要素ごとの演算が行われる。例えば、

```{code-cell} ipython3
a = np.array([1, 2])
b = np.array([3, 4])
print(a + b)
```

の例では、a と b の各要素に対して、和が取られていることが分かる。これは、他の四則演算に対しても同様である。

+++

### 各要素への関数適用

+++

NumPy には`numpy.exp`などのように、ベクトル・行列の各要素に対して、特定のスカラに対する演算を行う関数が用意されている。一例として、`numpy.sin`、`numpy.cos`、`numpy.exp`、`numpy.log`などがあり、高校レベルで思いつくものであれば大抵は用意されている。

+++

### ベクトルの内積

+++

内積の計算には二つのベクトルに対して`numpy.dot`を用いれば良い。

```{code-cell} ipython3
a = np.array([1, 2])
b = np.array([3, 4])
print(np.dot(a, b))
```

また、`numpy.dot`と同じ効果を現す演算子として`@`が用意されており、上と同様のコードは`@`を使って以下のように書ける。

```{code-cell} ipython3
print(a @ b)
```

### ベクトルの外積

+++

外積の計算には二つのベクトルに対して`numpy.cross`を用いれば良い。

```{code-cell} ipython3
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.cross(a, b))
```

なお、この`numpy.cross`は二次元ベクトルに対しても計算が可能で、その場合は二次元ベクトルを並べて作られる 2x2 行列の行列式が計算される。

```{code-cell} ipython3
a = np.array([1, 2])
b = np.array([3, 4])
print(np.cross(a, b))
```

この計算は二次元平面において a → b というベクトルの移動が時計回りの回転(負の値になる)なのか、反時計回りの回転(正の値になる)なのかを調べる時などに役に立つ。

+++

### 行列の転置

+++

行列の転置は NumPy の 2 次元配列を表わす変数に対して `.T`をつけることで計算できる。

```{code-cell} ipython3
A = np.array([[1, 2], [3, 4]])
print('A:\n', A)
print('A^T:\n', A.T)
```

また、3 以上の次元を持つ配列の場合には、`np.transpose`を用いることで、次元を自由な順序で入れ替えることができる。

```{code-cell} ipython3
A = np.arange(24).reshape((2, 3, 4))
print('Shape of A:', A.shape)  # 2x3x4のテンソル
A = np.transpose(A, axes=(2, 0, 1))  # 元の次元を2番目、0番目、1番目の順で入れ替え
print('Shape of A:', A.shape)  # 4x2x3のテンソル
```

## NumPyによる線形代数

+++

### 逆行列・疑似逆行列

+++

とある行列が正則行列 (逆行列を持つ = 行列式が 0 でない)ときには`numpy.linalg.inv`を用いて逆行列が計算できる。

```{code-cell} ipython3
A = np.array([[1, 2], [3, 4]])
invA = np.linalg.inv(A)
print(invA)
```

なお、その行列が正則行列かどうかを調べるには、行列のランクや行列式が 0 でないかを調べれば良いだろう。

```{code-cell} ipython3
# 行列のランク
np.linalg.matrix_rank(A)  # 2
```

```{code-cell} ipython3
# 行列式
np.linalg.det(A)
```

では、次に、正則でない行列 (= 特異行列という)の場合に逆行列を求めようとするとどうなるかを見てみる。

```{code-cell} ipython3
A = np.array([[1, 2], [2, 4]])
try:
    np.linalg.inv(A)
except Exception as e:
    print(f'Exception: {type(e).__name__:s}')
    print(f'Message: {str(e):s}')
```

上記の例では、例外処理をしており、**LinAlgError**が発生して、その例外メッセージが**Singular matrix**となることが確認できる。実際、上記の行列のランクは 1 で行列式は 0.0 になるので、各自で調べて見てほしい。

さて、このようは特異行列に対して、逆行列を求めるにはどうすれば良いのだろうか？特異行列に対して、逆行列を求めたい場面というのは実用上はそれなりに多く、その一番身近な例が、以下のような最小二乗問題を解きたい場合だろう。

+++

$$
\begin{equation}
\min_{\mathbf{x}} \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
\end{equation}
$$ (eq:least-squares)

+++

この場合、 $\mathbf{x}$ について被最小化関数を微分したものがゼロベクトルになる箇所を探せば、それが求める $\mathbf{x}$ だが、この際、被最小化関数の $\mathbf{x}$ についての微分によって得られる

$$
\mathbf{A}^\top \mathbf{Ax} = \mathbf{A}^\top \mathbf{b}
$$

という方程式において $\mathbf{A}^\top \mathbf{A}$ が特異行列になることは実問題においては頻繁に起こる問題である。

このような場合、上記の最小化問題の解は、特異行列に対して逆行列と似た性質を持つ疑似逆行列 (正しくは Moore-Penrose の疑似逆行列と呼ぶ)を右辺に乗ずることで求められる。

**疑似逆行列を $(\mathbf{A}^\top \mathbf{A})^\dagger$ のように表す**時、最小化を実現する $\mathbf{x}$ は次の式のように定まる。

$$
\mathbf{x} = (\mathbf{A}^\top \mathbf{A})^\dagger (\mathbf{A}^\top \mathbf{b})
$$

では、疑似逆行列とは、一体どういう性質を持った行列なのだろうか？

これを説明するには特異値分解について説明をする必要があるため、詳しくは触れないが、ぜひ NumPy で疑似逆行列を求める`numpy.linalg.pinv`を用いて、以下の計算を試してみてほしい。

```{code-cell} ipython3
# この行列は特異行列
A = np.array([[1, 2], [2, 4]])
pinvA = np.linalg.pinv(A)
print('- pinv(A) @ A')
print(pinvA @ A)
print('')
print('- A @ pinv(A) @ A')
print(A @ pinvA @ A)
```

::::{admonition} 問題
:class: question

上記のサンプルコードを踏まえて、正則行列に対する逆行列と、特異行列に対する疑似逆行列の違いと共通点について述べよ。

::::

+++

::::{admonition} 問題
:class: question

疑似逆行列を用いて、{eq}`eq:least-squares`を解く場合、$\mathbf{Ax} = \mathbf{b}$という線形問題が、不良設定である(未知数の数が式の数より多い)のか、過剰拘束である(未知数の数が式の数より少ない)のかによって、得られる解の意味が異なる。それぞれは、どのような解を与えるのだろうか。

::::

+++

#### 数値誤差について

上記の例ではNumPyの関数を用いて行列式や逆行列を求める方法について紹介してきた。前述の通り、行列が正則であれば、`numpy.linalg.inv`で逆行列が求められ、そうでなければ特異行列である旨のエラーメッセージが表示されることを確認したが、実際の数値計算で、入力された行列が正則か特異かを判断するのには若干の問題がある。

コンピュータによる数値計算では、数学的には特異である(＝行列式が0である)ような行列が、数値誤差の影響で非常に小さな非ゼロの行列式を持つために正則と判断されてしまうことがある。そのため、実際の問題で逆行列を計算するときには、注意が必要である。多くの場合は、逆行列の代わりに疑似逆行列を用いることで上記の問題を回避できるが、その場合には疑似逆行列を用いて得られる解が、どのような性質を持つかに留意する必要がある。

疑似逆行列を用いる、ということは行列が特異、すなわちランク落ちしている場合であるから、実際の線形方程式を満たす解は無数に存在することになる。疑似逆行列を用いて得られる解は、そのうち、ノルムの大きさ (より正確にはl2ノルムの大きさ)が最小になるものになるので、自身が求めたい解がその解で良いのかは、実際の問題を扱う上では重要だろう。

+++

::::{admonition} 練習問題
:class: question

数値誤差が行列式の計算に与える影響を調べるために、以下の行列の行列式を計算してみよ。

$$
\begin{pmatrix}
  100000 & 100001 & 100002 \\
  200000 & 200004 & 200002 \\
  300000 & 300003 & 300006 \\
\end{pmatrix}
$$

上記の行列は、各行がほとんど同じ値を取ってはいるが、解析的には行列式は0にはならない。
::::

+++

### 線形方程式を解く

+++

先ほどの説明では、 $\mathbf{Ax} = \mathbf{b}$ という線形方程式に対して、逆行列を計算して解 $\mathbf{x}$ を求める方法について述べた。

一方で、この方法は数値計算の側面からは、常に最良のやり方であるとは言い難い。通常、行列のサイズが $N \times N$ の時に、逆行列を求めるためのアルゴリズムには[Gauss の消去法](https://ja.wikipedia.org/wiki/%E3%82%AC%E3%82%A6%E3%82%B9%E3%81%AE%E6%B6%88%E5%8E%BB%E6%B3%95)などがあり、これらのアルゴリズムの計算量は $O(N^3)$ である。それに加え、逆行列をベクトル $\mathbf{b}$ に乗ずる操作 (計算量は $O(N^2)$ ) が必要となる。

一方で、同様に Gauss の消去法を使うにしても、逆行列を求めるのではなく、線形方程式を直接解くように用いることもできる。この場合も計算量は変わらず $O(N^3)$ なのだが、この場合は、直接的に解となる $\mathbf{x}$ が得られるため、逆行列をベクトル $\mathbf{b}$ に乗ずる計算が不要な分、効率が良い。

加えて、このような余計な行列計算を減らすことで、**浮動少数の計算による計算誤差を防ぐことができる**ため、より高精度に解が求められる。したがって、逆行列それ自体が不要であるならば、行列分解を用いて効率的に線形問題を解く`numpy.linalg.solve`を使う方がより適切である。

```{code-cell} ipython3
A = np.array([[1, 2], [3, 4]])
b = np.array([1, 1])
x = np.linalg.solve(A, b)
print(x)
```

また、実用的には $\mathbf{Ax} = \mathbf{b}$ という線形方程式で、 $\mathbf{A}$ は変わらないけれども $\mathbf{b}$ が異なるような問題を繰り返し解きたい場合というのが多くある。

`numpy.linalg.solve`は内部的には $O(N^3)$ の計算量をかけて行列を[LU 分解](https://ja.wikipedia.org/wiki/LU%E5%88%86%E8%A7%A3)した後、分解によって得られた下三角行列と上三角行列に対して線形問題を解く。この際、三角行列を係数に持つ線形方程式を解くための計算量は、一般の行列を係数に持つ線形問題を解くよりも小さく $O(N^2)$ であることに注意してほしい。ならば、何度も `numpy.linalg.solve`の内部で同じ行列に対して、何度も$O(N^3)$の計算量をかけて LU 分解するのは非効率的である。

このような場合、一度 LU 分解を計算してしまって、三角行列に対する線形問題を解く、という方法が考えられる。残念ながら、この方法は NumPy には実装されていないが類似ライブラリである SciPy を用いると、以下の形で実現できる。

```{code-cell} ipython3
import scipy as sp
import scipy.linalg
```

```{code-cell} ipython3
A = np.array([[1, 2], [3, 4]])
lu, piv = sp.linalg.lu_factor(A)
```

この結果において`lu`は上三角行列とした三角行列を結合した行列であり、`piv`はピボット選択によって生じた行の入れ替わり (連立方程式において方程式の順序を変える操作に対応)を表わす自然数の配列である。ピボット選択については、数値計算の初等的な教科書にも書かれている内容なので、各自、教科書を読むなどして勉強してみてほしい。

通常、LU 分解を行うときには、上三角行列、あるいは下三角行列のいずれかの対角成分が全て 1 になるように分解が計算されるため、本来、三角行列が 2 つなら$N(N+1)$要素が必要なところ$N^2$個の要素を 1 つの行列として表わしていることに注意すること。

```{code-cell} ipython3
print(lu)
```

この LU 分解の結果を用いて線形問題を解くには`sp.linalg.lu_solve`を用いる。実際のコードは以下の通り。

```{code-cell} ipython3
b1 = np.array([1, 1])
x1 = sp.linalg.lu_solve((lu, piv), b1)
print(x1)
```

```{code-cell} ipython3
b2 = np.array([2, 2])
x2 = sp.linalg.lu_solve((lu, piv), b2)
print(x2)
```

::::{admonition} 問題
:class: question

逆行列を`numpy.linalg.inv`によって求めることで線形問題を解く方法と、`numpy.linalg.solve`を用いて LU 分解により解を求める方法で計算精度がどの程度変化するかを確かめてみよ。

なお、より結果を分かりやすくするために`numpy.random.random`を用いて 100x100 および 100x1 程度の大きめのサイズで $\mathbf{A}$, $\mathbf{b}$ を作成し、精度については $\| \mathbf{A} \mathbf{x} - \mathbf{b} \|^2$ の大きさを使って調べること。
::::

+++

#### 直接法と反復法

先ほど紹介した`numpy.linalg.solve`はLU分解を用いて線形問題を解いていると説明した。このように行列分解などを用いて、線形問題の解を行列のサイズのみに依存する計算量で得るような解法を**直接法**と呼ぶ。直接法にはLU分解を用いる方法の他、[QR分解](https://ja.wikipedia.org/wiki/QR%E5%88%86%E8%A7%A3)など、別の行列分解を用いる方法がある。

一方、ここでは紹介しなかったが、線形問題の両辺に特定の線形代数的操作を施すことで、適当な $\mathbf{x}$ の初期値を徐々に線形問題の解に反復計算によって近づける方法を**反復法**と呼ぶ。反復法の中で最も単純なものは[Jacobi法](https://ja.wikipedia.org/wiki/%E3%83%A4%E3%82%B3%E3%83%93%E6%B3%95)がある (Jacobi法には固有値を求める手法などもありややこしい...)。Jacobi法は係数行列 $\mathbf{A}$ の対角成分だけを取りだした対角行列 $\mathbf{D}$を用いて、 $k$ 回反復時に得られている $\mathbf{x}^k$ を次の式で $\mathbf{x}^{k+1}$ に更新する。

$$
\mathbf{x}^{k+1} = \mathbf{D}^{-1} \left( \mathbf{b} - (\mathbf{A} - \mathbf{D}) \mathbf{x}^k \right)
$$

これ以外にも、Jacobi法を改良したGauss-Seidel法やSOR法がある他、実用的には[共役勾配法](https://ja.wikipedia.org/wiki/%E5%85%B1%E5%BD%B9%E5%8B%BE%E9%85%8D%E6%B3%95) (CG法)と呼ばれる、より収束の早いアルゴリズムが用いられる。最も一般的な共役勾配法は対称正定値行列 (正定値 = 全ての固有値が0より大きい)にしか用いることができないが、これを非対称行列に拡張した双共役勾配法 (BiCG法)など、多くの発展的な手法が存在する。

+++

::::{admonition} 問題
:class: question

前述の(反復計算による線形問題解法としての)Jacobi 法により、正しく線形問題が解ける(= `numpy.linalg.solve`と解が一致)ことを調べよ。なお、ヤコビ法は優対角行列と呼ばれる対角成分が、各列の対角成分以外の成分の総和より大きい行列にしか用いることができないので、そのような行列を各自適当に設定すること。
::::

+++

### 行列の固有値・固有ベクトル

+++

行列の固有値および固有ベクトルは、[主成分分析](https://ja.wikipedia.org/wiki/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)のような多次元データの分析の他、様々なデータの性質を分析する上で欠かせない情報である。

NumPy で行列の固有値ならびに固有ベクトルを求めるには`numpy.linalg.eig`を用いれば良い。

```{code-cell} ipython3
A = np.array([[1, 2], [2, 1]], dtype='double')
eigval, eigvec = np.linalg.eig(A)
print(eigval)
print(eigvec)
```

なお、通常、固有値に加えて、固有ベクトルを求めるには追加の計算が必要になるため、固有値だけが必要になる場合には、より高速に動作する関数として`numpy.linalg.eigvals`を使うのが良い。

また、上記の例では、あえて実数が固有値、固有ベクトルとなるように実対称行列の固有値・固有ベクトルを求めたが、通常、対称行列、ないし Hermite 行列の固有値・固有ベクトルの計算は非対称、ないし非 Hermite 行列の固有値よりも計算が簡単である。そのため、固有値の計算を行いたい対称が対称行列や Hermite 行列であることが分かっている場合には`numpy.linalg.eigh`や`numpy.linalg.eigvalsh`のように末尾に"h"のついている関数を使うと、より効率的である (とは言っても、効果を実感できるのは少なくとも 1000x1000 程度の行列からだろう)。

固有値を求めるという問題は、実用的に非常に広い応用を持つため、固有値を求める対象の行列の性質 (対称行列か、密行列か、疎行列か etc.)に応じて、様々なアルゴリズムが提案されている。ここでは、比較的単純なアルゴリズムとして Jacobi 法 (先ほどの線形問題を解く方法とは異なる)と QR 法について紹介する。

+++

#### Jacobi 法

+++

**Jacobi 法**は**対称行列**に対するアルゴリズムで、Givens 回転と呼ばれる行列の一部要素だけを変換するような下記の行列 $\mathbf{G}$ を用いる (値の入っていない箇所の要素は全て 0)。

$$
\mathbf{G} = \begin{pmatrix}
  1 &        &             &        &              &        &   \\
    & \ddots &             &        &              &        &   \\
    &        & \cos \theta & \cdots & -\sin \theta &        &   \\
    &        & \vdots      & \ddots & \vdots       &        &   \\
    &        & \sin \theta & \cdots &  \cos \theta &        &   \\
    &        &             &        &              & \ddots &   \\
    &        &             &        &              &        & 1 \\
\end{pmatrix}
$$

このとき、固有値を求めたい行列 $\mathbf{A}$ の $i$ 行 $j$ 列の要素を $a_{ij}$ と記すことにすると、上記の Givens 回転の回転量 $\theta$ を

$$
\tan 2\theta = \frac{-2a_{ij}}{a_{ii} - a_{jj}}
$$

となるように設定すると、

$$
\mathbf{A}' = \mathbf{G} \mathbf{A} \mathbf{G}^\top
$$

という変換を施すことで、 $a_{ij} = a_{ji}$ の要素を 0 にすることができる。この操作を行列 $\mathbf{A}$ の非対角成分のうち最も絶対値が大きい物が十分小さくなるまで繰り返すことで、固有値を対角行列とする行列 $\pmb{\Lambda}$ へと $\mathbf{A}$ を変換することができる。

また、Givens 回転行列は直交行列であるため、元の行列 $\mathbf{A}$ を $\pmb{\Lambda}$ に変換する過程で用いた全ての Givens 回転の積として得られる行列の各列が固有ベクトルとなる。

+++

#### QR 法

+++

**QR 法**は QR 分解を用いて一般的な行列 (= 非対称行列でもよい)の固有値・固有ベクトルを求めるアルゴリズムで、対象の行列 $\mathbf{A}$ が、QR 分解によって $\mathbf{A} = \mathbf{QR}$ (ただし、Q は直交行列、R は上三角行列)と分解されるとすると、

$$
\mathbf{A}' = \mathbf{Q}^\top \mathbf{A} \mathbf{Q} = \mathbf{Q}^\top \mathbf{Q} \mathbf{RQ} = \mathbf{RQ}
$$

のように行列 $\mathbf{Q}$ と $\mathbf{R}$の順序を入れ替えるような操作を繰り返すと、 $\mathbf{A}$ が徐々に固有値を対角成分に持つような対角行列 $\pmb{\Lambda}$ に近づいていく。この際、QR 分解の性質から行列 $\mathbf{Q}$ は直交行列であるから、上記の変換の過程で得られる全ての直交行列 $\mathbf{Q}$ の積を取ることで、固有ベクトルを各列に持つような直交行列を得ることができる。

+++

::::{admonition}　問題
:class: question

NumPy には多項方程式を解くための関数として`numpy.roots`という関数が用意されている ([参考](https://numpy.org/doc/stable/reference/generated/numpy.roots.html)、特性方程式の係数は手計算で求めること)。

この関数を用いて、以下の行列について、特性方程式を解くことで得られる固有値と、`numpy.linalg.eig`を解くことで得られる固有値が一致することを調べよ。

$$
\mathbf{A} = \begin{pmatrix}
  1 & 2 & 3 \\
  2 & 2 & 3 \\
  3 & 3 & 3
\end{pmatrix}
$$
::::

+++

::::{admonition} 問題
:class: question

上記の 3x3 行列 $\mathbf{A}$ に対して(固有値を求めるアルゴリズムとしての)Jacobi 法ならびに QR 法を用いて正しく固有値が求まる (= `numpy.linalg.eig`と同じ結果が得られる)ことを確かめよ。
::::

+++

## 疎行列

+++

疎行列とは、行列の要素のほとんどが 0 であるような行列を指す。計算機科学的には、$N \times N$の行列同士のかけ算には単純には$O(N^3)$の計算量がかかるが、ほとんどの要素が 0 であるのなら、その要素とのかけ算は無視できるので、計算を効率化できる。

疎行列の表し方にはいくつかの形式があるが、代表的なものは CSR(=Compressed Sparse Row)形式と CSC(=Compressed Sparse Column)形式の 2 つである。前者の CSR 形式は、各列が非ゼロ要素の配列として表わされ、そのの配列として行列が表わされている。一方、後者の CSC 形式は各行が非ゼロ要素の配列で表わされ、その配列として行列が表わされている。

一例として

$$
\begin{pmatrix}
1 & 0 & 0 \\
0 & 2 & 3 \\
4 & 0 & 5
\end{pmatrix}
$$

のような行列であれば、CSR 方式では

```python
csr = [
    [(0, 1.0), (2, 4.0)],  # 1列目
    [(1, 2.0)],  # 2列目
    [(1, 3.0), (2, 5.0)],  # 3列目
]
```

のような形で、CSC 方式では、

```python
csc = [
    [(0, 1.0)],  # 1行目
    [(1, 2.0), (2, 3.0)],  # 2行目
    [(0, 4.0), (2, 5.0)],  # 3行目
]
```

のような形で表せる。

+++

疎行列を扱うには SciPy の`scipy.sparse`モジュールを用いる。このモジュールには`csr_matrix`や`csc_matrix`という関数が用意されており、それぞれ CSR 方式・CSC 方式の疎行列を作成してくれる。作成方法は共通で、非ゼロ要素の行番号の配列`rows`、列番号の配列`cols`、そして非ゼロ要素そのものの配列`values`を引数として以下のように与えれば良い。

```{code-cell} ipython3
import scipy as sp
import scipy.sparse

rows = [0, 1, 1, 2, 2]
cols = [0, 1, 2, 0, 2]
values = [1.0, 2.0, 3.0, 4.0, 5.0]
m = sp.sparse.csr_matrix((values, (rows, cols)))

print(m.todense())  # 密行列として要素を表示
```

疎行列の場合にも`sp.sparse`や`sp.sparse.linalg`モジュールの関数を使うことで、行列に対する様々な演算ができるので、ぜひどのような計算ができるか各自で調べて見てほしい。

```{code-cell} ipython3

```
