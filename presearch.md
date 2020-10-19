## 功能
### 标注函数内部可用不同方法
- Keyword searches: looking for specific words in a sentence
- Pattern matching: looking for specific syntactical patterns
- Third-party models: using an pre-trained model (usually a model for a different task than the one at hand)
- Distant supervision: using external knowledge base
- Crowdworker labels: treating each crowdworker as a black-box function that assigns labels to subsets of the data
#### 提供各种统计功能
#### 可用于模型训练数据的迭代
#### 是一个良好的标注框架

#### 但是
感觉不太适合序列标注(还需更多调研)

## 劣势
是一个基于专家知识的自动弱(预)标注，有一定(甚至更高)的标注错误率
### 解决办法:
标注完之后再人工check(至少test/validate数据集需要再经过人工核验一遍)

## 主要优势
- 1. 将标志任务转化为核验任务，的确能减少手工标注工作量，
但效果取决于标注函数的逻辑是否有效
- 2. 支持基于多平台的数据处理(pandas, pyspark)
- 3. 支持多平台的模型训练(tf, torch)
