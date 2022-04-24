# OpenAi Gym

## Installation
OpenAi Gym 再windows+pip的安装一直有问题， 我建议用conda安装

```bash
conda install -c conda-forge gym-all
```

## Space Class
不管是agent的行为（action），亦或是agent的观测（observation），他们都可以用一个张量表示，gym把这类张量写在了一个叫做Sample的类里，这其中有两个继承了Sample的子类最为重要。

#### Space.sample()
在空间内随机抽样
#### Space.contains(x)
查看x是否再空间中
### Discrete
Discrete类里面储存了n个互斥的数。每一次只能sample其中一个数。

### Box
储存了一个n维张量，定义了最大数，最小数以及张量的shape。

## 写一个可以生成随机action的agent
详见CartPole.py文件

