# 1. 朴素贝叶斯的原理
贝叶斯为了解决一个叫“逆向概率”问题写了一篇文章，尝试解答在没有太多可靠证据的情况下，怎样做出更符合数学逻辑的推测。

“逆向概率”是相对“正向概率”而言。正向概率的问题很容易理解，比如我们已经知道袋子里面有 N 个球，不是黑球就是白球，其中 M 个是黑球，那么把手伸进去摸一个球，就能知道摸出黑球的概率是多少。但这种情况往往是上帝视角，即了解了事情的全貌再做判断。

在现实生活中，我们很难知道事情的全貌。贝叶斯则从实际场景出发，提了一个问题：如果我们事先不知道袋子里面黑球和白球的比例，而是通过我们摸出来的球的颜色，能判断出袋子里面黑白球的比例么？

贝叶斯原理与其他统计学推断方法截然不同，它是建立在主观判断的基础上：在我们不了解所有客观事实的情况下，同样可以先估计一个值，然后根据实际结果不断进行修正。
# 2. 朴素贝叶斯应用场景 
# 3. 朴素贝叶斯优缺点 
# 4. 朴素贝叶斯 sklearn 参数学习 
```angular2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import *
 
# 导入数据
train_data= pd.read_csv('F:\PY-Learning\CNEWS\cnews\cnews.train.txt', names=['title', 'content'], sep='\t', engine='python', encoding='UTF-8') # (50000, 2)
test_data = pd.read_csv('F:\PY-Learning\CNEWS\cnews\cnews.test.txt', names=['title', 'content'], sep='\t',engine='python',encoding='UTF-8') # (10000, 2)
val_data = pd.read_csv('F:\PY-Learning\CNEWS\cnews\cnews.val.txt', names=['title', 'content'], sep='\t',engine='python',encoding='UTF-8') # (5000, 2)
 
x_train = train_data['content']
x_test = test_data['content']
x_val = val_data['content']
 
y_train  = train_data['title']
y_test = test_data['title']
y_val  = val_data['title']
# print(y_val)
###################################################
#############处理样本#################################
 
## 默认不去停用词的向量化
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_train )
x_count_test = count_vec.transform(x_test )
 
## 去除停用词
count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train)
x_count_stop_test = count_stop_vec.transform(x_test)
 
## 模型训练
mnb_count = SVC()
mnb_count.fit(x_count_train, y_train)
mnb_count_y_predict = mnb_count.predict(x_count_test)
mnb_count.score(x_count_test, y_test)
 
## TF−IDF处理后在训练
## 默认配置不去除停用词
tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(x_train)
x_tfid_test = tfid_vec.transform(x_test)
 
## 模型训练
mnb_tfid = SVC()
mnb_tfid.fit(x_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
mnb_tfid.score(x_tfid_test, y_test)

```
# 5. 利用朴素贝叶斯模型结合 Tf-idf 算法进行文本分类