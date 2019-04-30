---
title: git 使用方法总结
date: 2019-04-30 18:27:19
tags: 
- git
categories: git
top: 20
---

# git 使用方法总结

比较基础的方法就不一一写了，碰到没使用过的在慢慢总结更新。

## git status

可以看到有三部分的内容，我们分为上中下。对于*Changes to be committed* 为暂存区，*Changes not staged for commit*为工作区，*Untracked files*为未添加文件（本地文件）。

先解释几个概念：

１．**工作区**：就是你在电脑里能看到的目录

２．**版本库**：工作区有一个隐藏目录`.git`，这个不算工作区，而是Git的版本库。Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的**暂存区**，还有Git为我们自动创建的第一个分支`master`，以及指向`master`的一个指针叫`HEAD`。

当我们想把文件往Git版本库里添加的时候，是分两步执行的：第一步是用`git add`把文件添加进去，实际上就是把文件修改添加到暂存区；第二步是用`git commit`提交更改，实际上就是把暂存区的所有内容提交到当前分支。因为我们创建Git版本库时，Git自动为我们创建了唯一一个`master`分支，所以，现在，`git commit`就是往`master`分支上提交更改。你可以简单理解为，需要提交的文件修改通通放到暂存区，然后，一次性提交暂存区的所有修改。

![图示](https://raw.githubusercontent.com/clancylian/blogpic/master/git.jpeg)

```bash
ubuntu@ubuntu-B250-HD3:~/Project/faceengine/faceengine$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        modified:   src/core/CMakeLists.txt
        modified:   src/core/algservice/FaceDetection/FaceDetector.cpp
        modified:   src/core/algservice/FaceDetection/FaceDetector.h
        modified:   src/core/algservice/FaceDetection/MTCNNCaffeDetector.h
        new file:   src/core/algservice/FaceDetection/MTCNNTensorrtDetector.cpp
        new file:   src/core/algservice/FaceDetection/MTCNNTensorrtDetector.h
        new file:   src/core/algservice/FaceDetection/resizeconvertion.cu
        modified:   src/core/scheduler/FaceEngine.cpp

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   src/core/CMakeLists.txt

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        patch
```

１．对于暂存区的内容，如果想要恢复和本地版本库一样，则需要输入命令：

```bash
$ git reset HEAD <file>...
$ git checkout -- <file>...
```

２．对于工作区的内容，如果需要恢复和本地版本库一样，则需要输入命令：

```bash
$ git checkout -- <file>...
```

３．如果想把工作区的内容或者本地内容添加到暂存区：

```bash
$ git add <file>...
```

４．当我们执行commit之后，想把版本回退到之前的版本：

```bash
# 回退到上一个版本
$ git reset --hard HEAD^
# 回退到上上版本
$ git reset --hard HEAD^^
# 回退到往上１００个版本
$ git reset --hard HEAD~100
```

如果想恢复到最新版本：

```bash
# 找到commit id
$ git reflog
9febc2e HEAD@{0}: pull: Fast-forward
7a028ee HEAD@{1}: clone: from https://xxx.git
# 回退到某个版本
$ git reset --hard 7a028ee
```

## git log

```bash
ubuntu@ubuntu-B250-HD3:~/Project/faceengine/faceengine$ git log
commit 9febc2e07ae0d5401dade4913578e2ae07381a73
Author: yeweijing <ypat_999@163.com>
Date:   Sun Apr 28 10:28:06 2019 +0800

    提交日志

commit ca94b89fd7bfe19e8891a0f5c5f05d1339b42480
Author: chendepin <danpechen@126.com>
Date:   Mon Apr 22 15:55:26 2019 +0800

    修复适应不同目录结构的问题

commit 0be1f9bb8f0d6fc41af30b88de70e539da421b67
Author: chendepin <danpechen@126.com>
Date:   Fri Apr 19 15:16:13 2019 +0800

    忽略 pyc 格式文件
```

**9febc2e07ae0d5401dade4913578e2ae07381a73**：为`commit id`（版本号），和SVN不一样，Git的`commit id`不是1，2，3……递增的数字，而是一个SHA1计算出来的一个非常大的数字，用十六进制表示为什么`commit id`需要用这么一大串数字表示呢？因为Git是分布式的版本控制系统，后面我们还要研究多人在同一个版本库里工作，如果大家都用1，2，3……作为版本号，那肯定就冲突了。**Author**：为作者。**Date**：提交日期。最后是日志内容。

## git rm

```bash
#　创建文件并提交
$ git add test.txt
$ git commit -m "add test.txt"

# 删除本地文件
$ rm test.txt

# 场景一
# 更新到暂存区
$ git rm/add test.txt
# 删除本地库文件
$ git commit -m "remove test.txt"

# 场景二
# 只删除本地文件，可以使用checkout复原
$ git checkout -- test.txt
```



## git diff

```bash
# 工作区与暂存区比较
$ git diff

# 暂存区与最新本地库比较
$ git diff --cached [<path>...]

# 工作区及暂存区与本地最新版本库比较
$ git diff HEAD [<path>...]

# 暂存区与指定commit-id比较
$ git diff --cached [<commit-id>] [<path>...]

# 比较两个commit-id之间的差异 
$ git diff [<commit-id>] [<commit-id>]

# 打补丁
# 将暂存区与版本库的差异做成补丁
$ git diff --cached > patch
# 将工作区以及暂存区与本地版本库的差异做成补丁
$ git diff HEAD > patch 
# 将工作区单个文件做成补丁
$ git diff <file>
```



## git stash

当我们在我们的分支上修改完成之后，需要先使用pull命令把远程库最新内容checkout下来，此时可能产生冲突导致pull不下来，因此需要现将我们修订的东西暂时储存起来。或者当我们在工作过程中，临时接到新任务，需要把当前的工作现场清理一下，等做完新任务后再恢复工作现场，这时候可以使用git stash来管理：

```bash
# 查看当前分支状态
$ git status
On branch dev
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    new file:   hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   readme.txt

# 将工作区和暂存区的文件保存到堆栈中
$ git stash
Saved working directory and index state WIP on dev: f52c633 add merge

# 恢复
$ git stash pop　#相当于使用git stash apply和git stash drop
On branch dev
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    new file:   hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   readme.txt

Dropped refs/stash@{0} (5d677e2ee266f39ea296182fb2354265b91b3b2a)

# 查看堆栈内容
$ git stash list

# 恢复指定的stash
$ git stash apply stash@{0}

# 清除stash
$ git stash clean
```

## 远程仓库管理

要关联一个远程库，使用命令如下，远程库的名字就是`origin`，这是Git默认的叫法，也可以改成别的，但是`origin`这个名字一看就知道是远程库。

```bash
$ git remote add origin git@github.com:clancylian/repo-name.git
```

关联后，由于远程库是空的，我们第一次推送`master`分支时，加上了`-u`参数，Git不但会把本地的`master`分支内容推送的远程新的`master`分支，还会把本地的`master`分支和远程的`master`分支关联起来，在以后的推送或者拉取时就可以简化命令。

```bash
$ git push -u origin master
```

此后，每次本地提交后，只要有必要，就可以使用以下命令提交：

```bash
$ git push origin master
```

从远程库克隆下来：

```bash
$ git clone https://github.com/clancylian/repo-name.git
```

```bash
# 查看远程库信息
$ git remote
origin

$ git remote -v
origin	https://github.com/amdegroot/ssd.pytorch.git (fetch)
origin	https://github.com/amdegroot/ssd.pytorch.git (push)

# 推送分支，master为本地分支
$ git push origin master

# checkout远程其他分支
$ git checkout -b dev origin/dev

#　如果push失败需要先pull下来
$ git pull
There is no tracking information for the current branch.
Please specify which branch you want to merge with.
See git-pull(1) for details.

    git pull <remote> <branch>

If you wish to set tracking information for this branch you can do so with:

    git branch --set-upstream-to=origin/<branch> dev
    
# 提示没有关联，需要先关联
$ git branch --set-upstream-to=origin/dev dev
Branch 'dev' set up to track remote branch 'dev' from 'origin'.
```

## 分支管理

```bash
# 创建分支并切换到分支
$ git checkout -b branch1
Switched to a new branch 'branch1'
# 相当于以下命令
$ git branch branch1
$ git checkout branch1
Switched to branch 'branch1'

# 查看当前分支,当前分支会有*号，也就是HEAD指向的分支
$ git branch
* branch1
  master
  
# 修改完分支内容之后，切换回主分支
$ git checkout master
Switched to branch 'master'

# 合并分支，git merge命令用于合并指定分支(branch1)到当前分支(master)
$ git merge branch1
Updating d46f35e..b17d20e
Fast-forward
 readme.txt | 1 +
 1 file changed, 1 insertion(+)
 
# 删除分支
$ git branch -d branch1
Deleted branch dev (was b17d20e).

######################################################3
# 当合并分支的时候出现冲突时
$ git status
On branch master
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)

You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

    both modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
# 查看文件内容出现
<<<<<<< HEAD
aaaaaaaaaaaaaaaaaaaaaaaa
=======
bbbbbbbbbbbbbbbbbbbbbbbb
>>>>>>> branch1

# 手动修改后
$ git add <file> 
$ git commit -m "conflict fixed"

```

##　标签管理

```bash
# 创建标签
$ git tag v1.0

$ git tag
v1.0

# 对历史版本打标签
$ git tag v0.9 f52c633

# 查看标签信息
$ git show v0.9

# 带有说明的标签
$ git tag -a v0.1 -m "version 0.1 released" 1094adb

# 删除本地标签
$ git tag -d v0.1
Deleted tag 'v0.1' (was f15b0dd)

# 提交标签到远程库
$ git push origin v1.0
Total 0 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
 * [new tag]         v1.0 -> v1.0

# 提交所有标签
$ git push origin --tags
Total 0 (delta 0), reused 0 (delta 0)
To github.com:michaelliao/learngit.git
 * [new tag]         v0.9 -> v0.9

# 删除远程标签
$ git tag -d v0.9
Deleted tag 'v0.9' (was f52c633)

$ git push origin :refs/tags/v0.9
To github.com:michaelliao/learngit.git
 - [deleted]         v0.9

```



## 参考链接

[git 学习网站推荐](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)

[gitignore 配置文件](https://github.com/github/gitignore)