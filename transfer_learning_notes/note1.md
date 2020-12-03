![image-20201203012807176](transfer_learning_pics/image-20201203012807176.png)

![image-20201203012833857](transfer_learning_pics/image-20201203012833857.png)

![image-20201203012855150](transfer_learning_pics/image-20201203012855150.png)

---

# 形式化

![image-20201203012917869](transfer_learning_pics/image-20201203012917869.png)

---

## 领域自适应问题

### 描述

![image-20201203012958021](transfer_learning_pics/image-20201203012958021.png)

特征的分布不一样，维度一样

![image-20201203013257690](transfer_learning_pics/image-20201203013257690.png)

### 解决

<img src="transfer_learning_pics/image-20201203013411097.png" alt="image-20201203013411097"  /> 

![image-20201203013545169](transfer_learning_pics/image-20201203013545169.png)

![image-20201203013621872](transfer_learning_pics/image-20201203013621872.png)

边缘分布反映整体分布

条件分布表示细致的形状

---

#### 边缘分布

![image-20201203013715544](transfer_learning_pics/image-20201203013715544.png)

令距离最小就完事了

这个“距离”是最大均值差异

扩展：

![image-20201203013806670](transfer_learning_pics/image-20201203013806670.png)

#### 条件分布适配

比较少

![image-20201203013846077](transfer_learning_pics/image-20201203013846077.png)

---

### 联合分布适配

两个都适配

问题：怎么获得条件分布？喏：

![image-20201203014003718](transfer_learning_pics/image-20201203014003718.png)

扩展：

![image-20201203014048362](transfer_learning_pics/image-20201203014048362.png)

JGSA是当时公开数据中最好的



联合分布适配的问题：

![image-20201203014246114](transfer_learning_pics/image-20201203014246114.png)

两个方面的权重

$\mu$是超参数？可训练参数？

![image-20201203014412142](transfer_learning_pics/image-20201203014412142.png)



##### 总结：

![image-20201203014456783](transfer_learning_pics/image-20201203014456783.png)

**深度学习+** 相对要好

#### 特征选择法

![image-20201203014602281](transfer_learning_pics/image-20201203014602281.png)

**SCL**

找出轴特征并进行对齐

#### 扩展：

![image-20201203014640830](transfer_learning_pics/image-20201203014640830.png)

加了很多项

或者把分类器也

![image-20201203014819279](transfer_learning_pics/image-20201203014819279.png)

这部分研究一般和别的方法结合

#### 子空间学习法

![image-20201203014911774](transfer_learning_pics/image-20201203014911774.png)

:warning:流形



![image-20201203015011050](transfer_learning_pics/image-20201203015011050.png)

![image-20201203015038352](transfer_learning_pics/image-20201203015038352.png)



##### 流形

在空间上把domain抽象成两个点，画一条最短距离（测地线）

取有限的点或无穷的点（积分

![image-20201203015154260](transfer_learning_pics/image-20201203015154260.png)

别的方法：

![image-20201203015212532](transfer_learning_pics/image-20201203015212532.png)

#### 总结

![image-20201203015301553](transfer_learning_pics/image-20201203015301553.png)

---

## 最新（2017）研究成果

![image-20201203015348100](transfer_learning_pics/image-20201203015348100.png)



![image-20201203015450568](transfer_learning_pics/image-20201203015450568.png)

---

没了
