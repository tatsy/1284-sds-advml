# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     formats: ipynb,md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""} tags=["remove-input"]
# (sec:pandas)=
#
# # Pandasの基本

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Pandas (panel-data-s の略)は、Python 向けのデータ分析ライブラリで、主にスプレッドシート状のデータを扱うことができる。Pandas では以下に説明する`DataFrame`を使って、データの操作やファイル入出力、グラフの作成等を行うことができる。

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
"""
下準備のコード
"""

import seaborn as sns
import matplotlib

# グラフの設定
rc = {"figure.dpi": 150}
sns.set_theme(style="white", palette="colorblind", rc=rc)
color_palette = sns.color_palette("colorblind")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## DataFrame の操作

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Pandas の中核をなすデータ構造に`DataFrame`がある。DataFrame とは、スプレッドシート状のデータを扱うデータ構造で、Excel のように数値を縦横に配置したデータを作ることができる。
#
# ```{image} https://pandas.pydata.org/docs/_images/01_table_dataframe.svg
# :align: center
# :width: 80%
# ```
#
# <div align="center">
#     
# (Pandasの[チュートリアル](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html)より引用)
#
# </div>

# %% [markdown] editable=true slideshow={"slide_type": ""}
# 例えば、テストの点数を集計したようなデータを考えてみる。以下、三人の学生について、数学と英語の点数を集計した物である (性別と点数の間には特別な意味はない)。

# %% editable=true slideshow={"slide_type": ""}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(
    {
        "Name": ["Taro", "Jiro", "Hanako"],
        "Math": [100, 90, 85],
        "English": [85, 90, 100],
    }
)
print(df)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# 上記の通り、`print`で`DataFrame`を出力すると、表のように整理された文字列が出力される。また、Jupyter 環境であれば、単に`df`と書くか、IPython モジュールの`display`を用いることで HTML により整形された表を出力することができる。

# %% editable=true slideshow={"slide_type": ""}
# 単にdfと書く
df

# %% editable=true slideshow={"slide_type": ""}
# displayの使用
from IPython.display import display

display(df)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# なお、`DataFrame`においては、各行にデフォルトで数字のインデックスが振られるが、これ自体は必要なく、例えば Name の列で各行を代表させれ十分であることも多い。このような場合には、DataFrame に対して`index`を指定して初期化する。

# %% editable=true slideshow={"slide_type": ""}
df_label = pd.DataFrame(
    {
        "Math": [100, 90, 85],
        "English": [85, 90, 100],
    },
    index=["Taro", "Jiro", "Hanako"],
)

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
df_label

# %% [markdown]
# ```{note}
# `index`を指定しない場合は、`index`に当たる0, 1, 2, ...がラベルであると見なされる。特に行お操作を行うときに、ラベルがインデックスなのか、その他の文字列等なのかを意識しておくことが大事になる。
# ```

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### 行と列の取り出し

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Pandas では行と列の意味合いが微妙に異なっており、それぞれを取り扱う場合に異なる操作が必要となる。
#
# **列の取り出し**
#
# `DataFrame`の各列は`Series`という型で表わされていて、df に各列のラベルを与えることで取り出すことができる。

# %% editable=true slideshow={"slide_type": ""}
col = df["Math"]
print('type is "{:s}"'.format(type(col).__name__))

# %% editable=true slideshow={"slide_type": ""}
col

# %% [markdown]
# また、複数のラベルを指定して、以下のように複数列を一度に取り出すこともできる。

# %%
cols = df[["Math", "English"]]
cols

# %% [markdown]
# **行の取り出し**
#
# 一方で、行を取り出す場合には`loc`あるいは`iloc`を用いる。`loc`は各行にラベルがついている場合に使用し、`iloc`は単純に行のインデックスを指定して使用する。

# %%
df.iloc[0]

# %%
df_label.loc["Taro"]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### 行と列の追加

# %% [markdown] editable=true slideshow={"slide_type": ""}
# **列の追加**
#
# 列を追加する方法はいくつかあるが、行と列でできるだけ似た操作を使うのなら`DataFrame`を辞書型のように扱って、データ列を代入する方法と`concat`を使う方法の 2 つがある。この 2 つなら辞書型として扱う方法の方が簡単で、行ラベルがインデックスなのか文字列等七日によって区別する必要がない。

# %% editable=true slideshow={"slide_type": ""} tags=["remove-input"]
# 辞書型として使う場合 (df自体が書き換わるので注意)
df_copy = df.copy()
df_copy["Physics"] = [75, 85, 80]
df_copy

# %% editable=true slideshow={"slide_type": ""}
# concatを用いる方法 (行ラベルがインデックス)
new_col = pd.Series({0: 75, 1: 85, 2: 80}, name="Physics")
pd.concat([df, new_col], axis=1)

# %% [markdown]
# `concat`を用いる場合、パラメータに`axis=1`を指定する (初期値は`axis=0` (行)に対応するので、明示的に`axis=1` (列)を指定する)。

# %%
# concatを用いる方法 (行ラベルが文字列)
new_col = pd.Series({"Taro": 75, "Jiro": 85, "Hanako": 80}, name="Physics")
pd.concat([df_label, new_col], axis=1)

# %% [markdown]
# **行の追加**
#
# 行を追加する場合も、上記の列の追加と同様に`df.loc`を辞書型と考えてデータ列を代入する方法 (こちらの方がシンプル)と、`concat`を用いてデータ行を結合する方法の 2 つがある。
#
# 行の場合は、行ラベルがインデックスなのか、文字列等なのかによって、書き方が異なる。特に、行を取り出す場合と異なり、`iloc`を用いるて行を追加することはできないので注意すること。

# %%
# locを用いる場合 (ラベルがインデックス)
df_copy = df.copy()
df_copy.loc[3] = ["Kikue", 100, 100]
df_copy

# %%
# locを用いる (ラベルが文字列他)
df_copy = df_label.copy()
df_copy.loc["Kikue"] = [100, 100]
df_copy

# %% [markdown]
# `concat`を使って「行」を追加する場合には、一度、行を含む`DataFrame`を作成して、それを`concat`で結合する必要がある。一行分のデータであれば、

# %%
# concatを用いる (ラベルがインデックス)
new_row = pd.Series({"Name": "Kikue", "Math": 100, "English": 100}, name=3)
pd.concat([df, pd.DataFrame(new_row).T], axis=0)

# %%
# concatを用いる (ラベルが文字列)
new_row = pd.Series({"Math": 100, "English": 100}, name="Kikue")
pd.concat([df_label, pd.DataFrame(new_row).T], axis=0)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### 行と列の削除

# %% [markdown] editable=true slideshow={"slide_type": ""}
# 行や列の削除には共通で`drop`を用いる。この関数の引数には`index=...`(行ラベルを指定)と`columns=...`(列ラベルを指定)という引数があり、これらを用いて削除すべき行や列を指定する。なお、`drop`は`DataFrame`自体を**更新しない**ので、もし`DataFrame`自体を更新したい場合には引数に`inplace=True`を与える。

# %% editable=true slideshow={"slide_type": ""}
df_copy = df_label.copy()
df_copy.drop(index=["Jiro"])

# %% editable=true slideshow={"slide_type": ""}
df_copy = df_label.copy()
df_copy.drop(columns=["Math"])

# %% [markdown] editable=true slideshow={"slide_type": ""}
# また、`labels=...`と`axis=...`を指定することで、行(`axis=0`)と列(`axis=1`)の何番目かを指定して削除することもできる。

# %%
df_copy = df_label.copy()
df_copy.drop(labels=["Taro"], axis=0)  # 行を削除

# %%
df_copy = df_label.copy()
df_copy.drop(labels=["Math"], axis=1)  # 列を削除

# %% [markdown]
# ### 要素へのアクセス

# %% [markdown]
# 要素へのアクセスには、これまでにも登場した`loc`や`iloc`を用いる。これらに行、列のラベルやインデックスを指定することで要素へのアクセスができる。

# %%
# ラベルを用いてアクセスする場合
print(df_label.loc["Taro", "English"])

# %%
# インデックスを用いてアクセスする場合 (0行: Taro, 1列: English)
print(df_label.iloc[0, 1])

# %% [markdown]
# また、要素のアクセス時は、通常の配列と同様に範囲を`:`を用いて指定することもできる。

# %%
# ラベルを用いて範囲指定
print(df_label.loc["Taro":"Hanako", "Math"])

# %%
# インデックスを用いて範囲指定
print(df_label.iloc[0:3, 0])

# %% [markdown]
# また、NumPy などと同様に、取り出したいインデックスやラベルを配列としても指定できる。

# %%
# ラベルを用いて指定
print(df_label.loc[["Taro", "Jiro"], "Math"])

# %%
# インデックスを用いて指定
print(df_label.iloc[0:2, 0])

# %% [markdown]
# 当然ながら、これらのアクセス方法を用いれば、`DataFrame`の値を書き換えることもできる。

# %%
df_copy = df_label.copy()
df_copy.loc["Taro", :] = 100  # Taroの全科目を100点に修正
df_copy

# %% [markdown]
# ## 数値計算

# %% [markdown]
# Pandas では、`DataFrame`の各行や各列、データ全体に対して統計量を簡単に取ることができる。統計量には平均(`mean`)や標準偏差(`std`)などが用意されており、以下のように計算できる。

# %%
# Taroの平均点を計算
print("Taro's avg: {:.3f}".format(df_label.loc["Taro"].mean()))
# Englishの平均点を計算
print("English avg: {:.3f}".format(df_label["English"].mean()))

# %% [markdown] editable=true slideshow={"slide_type": ""}
# これらを使うと、偏差値なども簡単に計算できる。

# %% editable=true slideshow={"slide_type": ""}
# 数学の偏差値を計算
math_dev = 50.0 + (df["Math"] - df["Math"].mean()) / df["Math"].std() * 10.0
math_dev.name = "Math dev."
print(math_dev)

# %% editable=true slideshow={"slide_type": ""}
# 英語の偏差値を計算
eng_dev = 50.0 + (df["English"] - df["English"].mean()) / df["English"].std() * 10.0
eng_dev.name = "Eng. dev."
print(eng_dev)

# %% [markdown]
# このようにして計算した結果を表に挿入することで、より多くの情報を含んだ`DataFrame`を作り上げていくことができる。

# %%
df_copy = df_label.copy()
df_copy.loc[:, math_dev.name] = math_dev.values
df_copy.loc[:, eng_dev.name] = eng_dev.values
df_copy

# %% [markdown]
# 行や列の順序を入れ替えたい場合には`loc`や`iloc`に入れ替え後のラベルやインデックスの配列を指定すれば良い。

# %%
df_copy = df_copy.loc[:, ["Math", "Math dev.", "English", "Eng. dev."]]
df_copy

# %% [markdown]
# 最後に Math と English のそれぞれについて平均点を追加してみる。

# %%
math_avg = df["Math"].mean()
eng_avg = df["Math"].mean()
df_copy.loc["Avg"] = [math_avg, "N/A", eng_avg, "N/A"]
df_copy

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## データの入出力

# %% [markdown]
# 次に、先ほど作成した`DataFrame`をファイルに出力してみよう。出力できるファイル形式は様々だが、CSV と Excel 形式のファイルをここでは試してみる。
#
# **ファイルへの出力**
#
# CSV を出力する場合には`DataFrame`の`to_csv`を用いれば良く、第 1 引数に出力先のファイル名を指定する。加えて**日本語を出力する場合には`encoding="utf_8_sig"`あるいは`encoding="shift_jis"`を指定**しておく。

# %%
# 最初の列の表記を変更
df_copy = df_label.rename(index={"Taro": "太郎", "Jiro": "次郎", "Hanako": "花子"})
df_copy

# %%
# CSVに出力
df_copy.to_csv("pandas.csv", encoding="utf_8_sig")

# %% [markdown]
# Excel ファイルを出力する場合は、`to_excel`に対して、出力ファイル名を含むいくつかの引数を指定する。Excel の場合は、エンコーディングを指定する必要はない (指定できない)。なお、Excel ファイルの操作を行う場合には、Pandas 以外に`openpyxl`をインストールしておく必要がある。

# %%
# Excelファイルに出力
df_copy.to_excel("pandas.xlsx")

# %% [markdown]
# 正しく出力されると、Excel 上で以下のように表の内容が確認できる。
#
# ```{image} ./imgs/pandas_to_excel.jpg
# :align: center
# :width: 90%
# ```

# %% [markdown]
# **ファイルの読み取り**
#
# ファイルからの読み取りには`read_csv`や`read_exel`といった関数を代わりに用いる。

# %%
df_csv = pd.read_csv("pandas.csv")
df_csv

# %%
df_excel = pd.read_excel("pandas.xlsx")
df_excel

# %% [markdown]
# ただし、上記の例ではファイル保存時に index が文字列となっているファイルを保存しているため、そのまま読み込むと、自動的に index が数字となっている列が追加されてしまう。これを防ぐためには index に相当する列が何列目なのかを`index_col=...`で指定すれば良い。

# %%
df_csv = pd.read_csv("pandas.csv", index_col=0)
df_csv

# %%
df_excel = pd.read_excel("pandas.xlsx", index_col=0)
df_excel

# %% [markdown]
# ## グラフの作成

# %% [markdown]
# `DataFrame`は`plot`というメンバを持ち、さらに`plot`に対してグラフの種類に対応するメソッドを呼び出すことで簡単にグラフを作成することができる。以下は棒グラフと散布図を作る例である。また、同様の出力は`plot`をメソッドとして呼び出して`kind`パラメータにグラフの種類を指定することでも実現できる。

# %% editable=true slideshow={"slide_type": ""}
# 再度DataFrameを作成
df = pd.DataFrame(
    {
        "Name": ["Taro", "Jiro", "Hanako"],
        "Math": [100, 90, 85],
        "English": [85, 90, 100],
    }
)

# %% editable=true slideshow={"slide_type": ""}
# 棒グラフの作成
df.plot.bar(x="Name", y=["Math", "English"])
plt.show()

# %% editable=true slideshow={"slide_type": ""}
# plot(..., kind=...)を使う場合
df.plot(x="Name", y=["Math", "English"], kind="bar")
plt.show()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# なお、グラフの見た目を調整したい場合には、`plt.title`等のメソッドを順次呼び出せば良い。

# %% editable=true slideshow={"slide_type": ""}
df.plot.bar(x="Name", y=["Math", "English"])
plt.title("Exam scores")
plt.xlabel("Student")
plt.ylabel("Score")
plt.ylim([0, 120])
plt.legend(loc="upper right")
plt.show()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# また、複数のグラフを並べたい場合には`plot`の`ax`パラメータに対して Matplotlib の`SubplotAxis`を指定すれば良い。

# %% editable=true slideshow={"slide_type": ""}
fig = plt.figure(figsize=(8, 4))

ax = fig.add_subplot(121)
df.plot(ax=ax, x="Name", y="Math", kind="bar", legend=False, color=color_palette[0])
ax.set_title("Math")

ax = fig.add_subplot(122)
df.plot(ax=ax, x="Name", y="English", kind="bar", legend=False, color=color_palette[1])
ax.set_title("English")

plt.tight_layout()
plt.show()

# %% editable=true slideshow={"slide_type": ""}
