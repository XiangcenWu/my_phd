# Progressive Neural Architecture Search

原文地址：[Progressive Neural Architecture Search](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)

这篇文章的第三部分对SEARCH SPACE进行了很好的概括

### Section 3: Architecture Search Space

文章的中心思想是先学习一个cell（其中包含很多layer以及skip connection），再把学习到的cell堆积起来形成最终的卷积神经网络

#### Section 3.1: Cell Topologies
