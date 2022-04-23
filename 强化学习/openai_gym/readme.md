# OpenAi Gym

## Installation
OpenAi Gym 再windows+pip的安装一直有问题， 我建议用conda安装

```bash
conda install -c conda-forge gym-all
```

## Space Class
不管是agent的行为（action），亦或是agent的观测（observation），他们都可以用一个张量表示，gym把这类张量写在了一个叫做Sample的类里，这其中有两个继承了Sample的子类最为重要。

### Discrete


### Box
