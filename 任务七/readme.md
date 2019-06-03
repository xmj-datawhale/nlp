# 1. pLSA、共轭先验分布；LDA主题模型原理
# 1.1 PLSA（Probabilistic Latent Semantic Analysis）
**概率隐语义分析**（PLSA）是一个著名的针对文本建模的模型，是一个生成模型。因为加入了主题模型，所以可以很大程度上改善多词一义和一词多义的问题。Hoffmm在1999年提出了概率隐语义分析。他认为每个主题下都有一个词汇的概率分布，而一篇文章通常由多个主题构成，并且文章中的每个单词都是由某个主题生成的。

关于PLSA的原理及公式推导可以参考博客 http://www.cnblogs.com/bentuwuying/p/6219970.html

1.1.1 PLSA的优势

定义了概率模型，而且每个变量以及相应的概率分布和条件概率分布都有明确的物理解释。
相比于LSA隐含了高斯分布假设，pLSA隐含的Multi-nomial分布假设更符合文本特性。
pLSA的优化目标是是KL-divergence最小，而不是依赖于最小均方误差等准则。
可以利用各种model selection和complexity control准则来确定topic的维数。

1.1.2pLSA的不足

概率模型不够完备：在document层面上没有提供合适的概率模型，使得pLSA并不是完备的生成式模型，而必须在确定document i的情况下才能对模型进行随机抽样。
随着document和term 个数的增加，pLSA模型也线性增加，变得越来越庞大。
EM算法需要反复的迭代，需要很大计算量。

## 1.2 共轭先验分布
设θ是总体分布中的参数(或参数向量)，π(θ)是θ的先验密度函数，假如由抽样信息算得的后验密度函数与π(θ)有相同的函数形式，则称π(θ)是θ的(自然)共轭先验分布。

1.2.1 共轭先验分布的参数确定
如对于总体为二项分布，其成功概率的共轭先验分布为Beta(α,β)Beta(α,β)，在确定了共轭先验分布之后，我们还需要确定先验分布中的参数，像这里的(α,β)(α,β)。因此下面介绍两种常见方法来确定其参数。

(1) 先验矩

假如利用先验信息能得到成功概率θθ的若干个估计值，θ1、θ2、...、θk θ1、θ2、...、θkθ1、θ2、...、θk。由此可算得先验均值$\overline{θ}$和先验方差$S^2_θ$。
同时由先验分布贝塔分布Beta(α,β)，可以得出(α,β)(α,β)表示的期望和方差。
由此可解得(α,β)(α,β)的值。

(2) 先验分位数

若由先验信息可以确定贝塔分布的两个分位数，则可由分位数的定义列出两个方程组同样接触所需参数。

## 1.2.2  常见的共轭先验分布

总体分布|	参数|	共轭先验分布
---:|---:|---:
二项分布|	成功概率|	贝塔分布$\B(\alpha,\beta)$
泊松分布|	均值|	伽马分布$\Gamma(k,\theta)$
指数分布|	均值的倒数|	伽马分布$\Gamma(k,\theta)$
正态分布(方差已知)|	均值|	正态分布$\N(\mu,\sigma^2)$
正态分布(方差未知)|	方差|	逆伽马分布$\IGa(\alpha,\beta)$

## 1.3 LDA主题模型原理
事实上，理解了pLSA模型，也就差不多快理解了LDA模型，因为LDA就是在pLSA的基础上加层贝叶斯框架，即LDA就是pLSA的贝叶斯版本（正因为LDA被贝叶斯化了，所以才需要考虑历史先验知识，才加的两个先验参数）。

对于语料库中的每篇文档，LDA定义了如下生成过程（generative process）：

(1).对每一篇文档，从主题分布中抽取一个主题

(2) 从上述被抽到的主题所对应的单词分布中抽取一个单词

(3) 重复上述过程直至遍历文档中的每一个单词。

之前没接触过，自己也没完全搞懂，就先不写这部分了，强烈推荐["LDA数学八卦"系列](http://www.52nlp.cn/lda-math-%E6%B1%87%E6%80%BB-lda%E6%95%B0%E5%AD%A6%E5%85%AB%E5%8D%A6)，内容详细通俗易懂。
# 2. LDA应用场景 
- 通常LDA用户进主题模型挖掘，当然也可用于降维。 
- 推荐系统：应用LDA挖掘物品主题，计算主题相似度 
- 情感分析：学习出用户讨论、用户评论中的内容主题
# 3. LDA优缺点 
LDA算法既可以用来降维，又可以用来分类，但是目前来说，主要还是用于降维。

LDA算法的主要**优点**有：

1）在降维过程中可以使用类别的先验知识经验，而像PCA这样的无监督学习则无法使用类别先验知识。

2）LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

LDA算法的主要**缺点**有：

1）LDA不适合对非高斯分布样本进行降维，PCA也有这个问题。

2）LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。

3）LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好。

4）LDA可能过度拟合数据。
# 4. LDA 参数学习 
```angular2
n_components : int, optional (default=10)
    主题数

doc_topic_prior : float, optional (default=None)
    文档主题先验Dirichlet分布θd的参数α

topic_word_prior : float, optional (default=None)
    主题词先验Dirichlet分布βk的参数η

learning_method : 'batch' | 'online', default='online'
    LDA的求解算法。有 ‘batch’ 和 ‘online’两种选择

learning_decay : float, optional (default=0.7)
   控制"online"算法的学习率，默认是0.7

learning_offset : float, optional (default=10.)
    仅在算法使用"online"时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
    
max_iter : integer, optional (default=10)
    EM算法的最大迭代次数

batch_size : int, optional (default=128)
   仅在算法使用"online"时有意义， 即每次EM算法迭代时使用的文档样本的数量。

evaluate_every : int, optional (default=0)
    多久评估一次perplexity。仅用于`fit`方法。将其设置为0或负数以不评估perplexity
     训练。
     
total_samples : int, optional (default=1e6)
    仅在算法使用"online"时有意义， 即分步训练时每一批文档样本的数量。在使用partial_fit函数时需要。

perp_tol : float, optional (default=1e-1)
    batch的perplexity容忍度。

mean_change_tol : float, optional (default=1e-3)
    即E步更新变分参数的阈值，所有变分参数更新小于阈值则E步结束，转入M步。

max_doc_update_iter : int (default=100)
    即E步更新变分参数的最大迭代次数，如果E步迭代次数达到阈值，则转入M步。

n_jobs : int, optional (default=1)
   在E步中使用的资源数量。 如果为-1，则使用所有CPU。
     ``n_jobs``低于-1，（n_cpus + 1 + n_jobs）被使用。

verbose : int, optional (default=0)
    详细程度。

```
# 5. 使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类
## 5.1 LDA生成主题特征
```angular2
from sklearn.decomposition import LatentDirichletAllocation
print('--------------------训练完成-----------------------')
# 利用已训练好的模型将doc转换为话题分布
doc_topic_dist = model.transform(x_train)
# 通过调用lda.perplexity(X)函数，可以得知当前训练的perplexity
print(doc_topic_dist, '当前训练的perplexity', model.perplexity(x_train), sep='\n')

def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print('打印主题-词语分布矩阵')
    return model.components_

tf_feature_names = vectorizer.get_feature_names()
m = print_top_words(model, tf_feature_names, 20)
print(m)
```
## 5.2 LDA + SVM 文本分类
```angular2
tf_vectorizer = CountVectorizer()
tf_train = tf_vectorizer.fit_transform(train_content)
tf_test = tf_vectorizer.fit_transform(test_content)```
lda = LatentDirichletAllocation(n_components=10,
                                    max_iter=20,
                                    learning_method='batch',
                                    evaluate_every=200,
                                    verbose=0)
x_train = lda.fit(tf_train).transform(tf_train)
x_test = lda.fit(tf_test).transform(tf_test)
clf = nb.SVC()
clf.fit(x_train, y_train)
print('--------------------训练完成-----------------------')
pred = clf.predict(x_test)
print("classification report on test set for classifier:")
print(classification_report(y_test, pred ))

```
* 参考
https://blog.csdn.net/nc514819873/article/details/89374542