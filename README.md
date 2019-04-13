## 多台电脑同步更新
首先，在这台电脑上可以 hexo d -g发布。
其次，为了保证你的更新能保持同步，每次更新文章之后，把文章的更改add并commit，然后git push。
之后，再其他电脑上git pull一遍即可。

## git clone source 分支
```
git clone -b source https://github.com/clancylian/clancylian.github.io
```

## 增加新博客后
```
git add .
git commit -m 'add xxxx'
git push origin source
```
