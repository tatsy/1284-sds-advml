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

(sec:notation)=
# 資料中の表記について

+++

## Pythonモジュールのエイリアス

+++

資料中では、特に指定がない場合、以下のようにモジュールがエイリアスを使ってインポートされているとする。従って、`numpy.array`などは、資料中で`np.array`のように略記することがあるので注意すること。

```{code-cell} ipython3
:tags: [remove-cell]

# ruff: noqa: I001
```

```{code-cell} ipython3
# NumPyなど前半で使うモジュール
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
```

```{code-cell} ipython3
# PyTorch系のモジュール
import torch.nn as nn
import torch.nn.functional as F
```

## 数学の記号

+++

本資料では、特に断りがない限り、以下のように数学のシンボルを区別して用いる。

+++

| 表記 | 意味 |
|:-----:|:-----|
| $x$ | スカラー |
| $\mathbf{x}$ | 列ベクトル |
| $x_{i}$ | ベクトル$\mathbf{x}$の$i$番目の要素 |
| $\mathbf{0}$ | 全ての要素が0の列ベクトル |
| $\mathbf{1}$ | 全ての要素が1の列ベクトル |
| $\mathbf{A}$ | 行列 |
| $A_{ij}$ | 行列$\mathbf{A}$の$i$行$j$列要素 |
| $\| \mathbf{A} \|$ | 行列$\mathbf{A}$の行列式 |
| $\mathbf{A}^{-1}$ | 行列$\mathbf{A}$の逆行列 |
| $\mathbf{A}^{\dagger}$ | 行列$\mathbf{A}$の疑似逆行列 |
| $\mathbf{I}$ | 単位行列 |
| $\mathbf{O}$ | 全ての要素が0の行列 |
| $\mathbf{x}^\top$ | ベクトル・行列の転置 |
| $\\| \mathbf{x} \\|$ | Euclidノルム |
| $\\| \mathbf{x} \\|_p$ | $\ell_p$ ノルム |
| $\\| \mathbf{A} \\|_F$ | Frobeniusノルム |
| $\mathbb{N}$ | 自然数体 |
| $\mathbb{Z}$ | 整数体 |
| $\mathbb{R}$| 実Euclid空間 |
| $\mathbb{R}^{n}$ | $n$次実Euclid空間 |
| $\mathcal{B}$ | Banach空間 |
| $\mathcal{H}$ | Hilbert空間 |
| $\langle f, g \rangle$ | Hilbert空間上での内積 |

```{code-cell} ipython3

```
