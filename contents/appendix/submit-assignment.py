# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: sdsadvml-NOwn6uOF-py3.10
#     language: python
#     name: python3
# ---

# %% [markdown]
# (sec:submit-assignment)=
# # 課題の提出方法

# %% [markdown]
# 本講義では、課題の提出と採点に[GitHub Classroom](https://docs.github.com/ja/education/manage-coursework-with-github-classroom/get-started-with-github-classroom/about-github-classroom)を使用する。
#
# GitHub ClassroomはGitのプッシュ機能を使って課題を提出し、GitHub Actionsの機能を使って課題を自動採点する。
#
# 各課題にはテンプレートファイルが用意されており、そのテンプレートを学生自身が編集してコミット、プッシュをすると、自動で採点プログラムが動き、採点結果を学生自身も確認できるのが特徴である。

# %% [markdown]
# ## 課題用リポジトリの作成

# %% [markdown]
# 各課題の招待リンクは、課題の直前に教員が指示する。
#
# その招待リンクからGitHub Classroomにアクセスすると、以下のような画面が表示されるので、**Accept this assignment**をクリックする。

# %% [markdown]
# :::{image} ./imgs/accept_assignment.jpg
# :align: center
# :class: image-stylish image-with-border
# :::

# %% [markdown]
# すると、以下のような画面が表示される。画面中央にあるGitHubのリンクが課題用のリポジトリのURLである。

# %% [markdown]
# :::{image} ./imgs/you_are_ready.jpg
# :align: center
# :class: image-stylish image-with-border
# :::

# %% [markdown]
# このURLにアクセスすると、以下のような課題用のテンプレートを含むリポジトリが現れる。

# %% [markdown]
# :::{image} ./imgs/assignment_repo.jpg
# :align: center
# :class: image-stylish image-with-border
# :::

# %% [markdown]
# 以後は、通常通り、このリポジトリのコードをクローンして、課題用のプログラムを作成した後、コミット・プッシュを行なうと、自動的に採点プログラムが動く。

# %% [markdown]
# ## SHA値の取得方法

# %% [markdown]
# 各受講生が採点をして欲しいコミットを明示するために、各コミットに振られたSHA値を用いる。Gitで管理されたコミットには識別用に必ずSHA値が設定されている。
#
# Gitで用いられているSHA値はSHA-1という種類でビット数にして160ビットの長さを持つハッシュの一種である。通常、4ビットを1単位として16進数の文字(0-9, a-f)が20文字の文字列として表現される。
#
# SHA値は、各自のコンピュータでGitのレポジトリをターミナル等で開き、`git log`を実行することで調べられる。一例として、以下のような出力が得られる。
#
# ```shell
# commit 12f434cc79f5d40b6a0d4e3a6e228e7eeeeb8354 (HEAD -> master, origin/master)
# Author: Tatsuya Yatagawa <xxxxx@xxxxx.com>
# Date:   Mon Oct 14 12:29:46 2024 +0900
#
#     Minor update.
#
# commit 744877b6feab88b101f2841a3ca772d0d8ec7882
# Author: Tatsuya Yatagawa <xxxxx@xxxxx.com>
# Date:   Mon Oct 14 12:24:15 2024 +0900
#
#     Minor update.
# ```
#
# このような出力において`commit`の右側に書かれている英数字の文字列がSHA値である。

# %% [markdown]
# また、SHA値はGitHub上でブラウザから確認することもできる。各レポジトリの画面右上の方にある「Commits」という文字をクリックしてみよう。
#
# :::{image} ./imgs/click_commits.jpg
# :align: center
# :class: image-stylish image-with-border
# :::
#
# すると、次のようなコミットの履歴画面が現れる。各コミットの右側にはSHA値の先頭6文字が表示されており、さらにその右にコピーボタンがある。このコピーボタンをクリックすると、SHA値全体がコピーできる。

# %% [markdown]
# :::{image} ./imgs/copy_sha.jpg
# :align: center
# :class: image-stylish image-with-border
# :::
