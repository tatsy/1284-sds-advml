機械学習発展 (実践)
===

[![GitHub Pages](https://github.com/tatsy-classes/1284-sds-advml/actions/workflows/gh-pages.yaml/badge.svg)](https://github.com/tatsy-classes/1284-sds-advml/actions/workflows/gh-pages.yaml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

講義の概要
---

この講義ではPythonを用いて、画像認識と機械学習の実践的な利用法について学ぶ。資料の中では、ライブラリを使った実装を用いる場合も、その背景にある理論とライブラリに頼らない簡易実装を与える。これにより、**ライブラリのユーザから卒業**し、自分の必要とする機械学習のメソッドを実装する力を養う。

### 求める前提知識

- Pythonの**読み書きが不自由なく**できる (基本的な読み書きだけだと辛いかもしれない)
- WindowsのPowerShellやMac/Linuxの**ターミナルの基本操作**ができる
- 大学1, 2年生レベルの初等的な数学(**特に線形代数**)の知識

### 想定受講者

- Pythonを用いた画像認識・機械学習システムの実装方法について学びたい人
- 深層学習だけでなく、画像認識・機械学習技術の背景にある理論を学びたい人

### 資料中の表記について

- [こちら](sec:notation)をご覧ください

### 参考図書

- M. Beyeler 著、池田ほか訳、『OpenCVとPythonによる機械学習プログラミング』、マイナビ [[URL]](https://book.mynavi.jp/ec/products/detail/id=92292)
- A. Geron 著、下田 監修、長尾 訳、『scikit-learn、Keras、TensorFlowによる実践機械学習 (第2版)』、オライリー社 [[URL]](https://www.oreilly.co.jp/books/9784873119281/)
- 赤穂 著、『カーネル多変量解析』、岩波書店 [[URL]](https://www.iwanami.co.jp/book/b257891.html)
- R. S. Sutton, A. G. Barto 著、鈴木ほか訳、『強化学習 (第2版)』、森北出版 [[URL]](https://www.morikita.co.jp/books/mid/082662)

---

以下、一橋大生向け情報

講義について
---

- 実施時限: 秋冬学期 第2限
- 教室: 東2号館棟2206教室

### 講義形式

- 資料の内容を全て講義内で解説することはしないので、必要に応じて自習をお願いします。
- 資料を全て理解する必要はありませんが、演習については課題が解けるように取り組んでください。

### 講義日程

- **事前準備**
  - [Python環境の設定](https://tatsy-classes.github.io/1284-sds-prog2/contents/setup-python.html)
  - [Git環境の設定](https://tatsy-classes.github.io/1284-sds-prog2/contents/setup-git.html)

- **第1回**
  - {ref}`sec:numpy`
  - {ref}`sec:matplotlib` (自習)
  - {ref}`sec:pandas` (自習)

- **第2回**
  - {ref}`sec:opencv`
  - {ref}`sec:figure-detection`

- **第3回**
  - {ref}`sec:scikit-learn`

- **第4回**
  - {ref}`sec:exercise-sudoku`

- **第5回**
  - {ref}`sec:exercise-sudoku`

- **第6回**
  - {ref}`sec:data-visualization`

- **第7回**
  - {ref}`sec:feature-extraction`
  - {ref}`sec:deep-learning` (自習)

- **第8回**
  - {ref}`sec:exercise-ogura`

- **第9回**
  - {ref}`sec:exercise-ogura`

- **第10回**
  - {ref}`sec:reinforcement-learning`
  - {ref}`sec:q-learning`
  - {ref}`sec:deep-q-learning` (自習)

- **第11回**
  - {ref}`sec:othello-player`

- **第12回**
  - {ref}`sec:exercise-othello`

- **第13回**
  - {ref}`sec:exercise-othello`

課題の提出方法
---

- [こちら](sec:submit-assignment)を参照ください。

Build Status
---

```{nb-exec-table}
```
