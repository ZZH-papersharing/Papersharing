1.Graph Contrastive Learning with Augmentations
对原始输入图进行augmentation,文章选用Node dropping、Edge perturbation、Attribute masking、 Subgraph对图进行增强，通过对比学习训练出一个encoder f(·)，目的是为了是原始图经过f(·)后能够提取出最重要的信息，以便用于下游任务。

2.Adversarial Graph Augmentation to Improve Graph Contrastive Learning
文章选择了Edge-dropping 方式对图进行增强，在进行Edge-dropping augmentation时，通过一个可学习的GNN网络产生的一对word embedding来判断边是否需要被drop。后续进行对比学习时，是将原图和经过增强后的图对比，以找到最大的mutual information。

3. idea
可以选择将第二篇文章中的输入的原始图像替换成另外一个可学习的GNN进行Augmentation, 同时引入正则项，让两个GNN不会向同一个方向学习发展。

4. 需要实现的方向：
在谱域上进行图对比学习，让两个方向的增强都是可以学习训练的，需要看一下相关文章！
补充1： 可以尝试一下与下游任务结合


other： 1：使用强化学习证明levy walks 是最高效的walk方式  Optimal foraging theory
        2：通过生成式来生成食物网。
