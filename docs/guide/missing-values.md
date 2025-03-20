# 缺失值处理

## 知识要点

### 理论知识
- 缺失值类型识别
- MCAR、MAR、MNAR区别
- 缺失值处理策略
- 缺失值影响评估

### 实践要点
- 缺失值可视化分析
- 删除法实现
- 填充法应用
- 高级填充方法

## 缺失值类型

### 1. 完全随机缺失（MCAR）
- 数据的缺失完全是随机的
- 缺失与其他变量无关
- 例如：随机抽样过程中的遗漏

### 2. 随机缺失（MAR）
- 数据的缺失与其他观察变量相关
- 与缺失变量本身的值无关
- 例如：高收入人群倾向于不填写收入信息

### 3. 非随机缺失（MNAR）
- 数据的缺失与缺失值本身相关
- 最难处理的缺失类型
- 例如：成绩差的学生倾向于不填写成绩

## 处理方法

### 1. 删除法
   - 列删除
   - 行删除
   ```python
删除含有缺失值的行
df.dropna()
删除缺失值过多的列
df.dropna(axis=1, thresh=df.shape[0]0.5)
```
2. 填充法
   - 均值填充
   - 中位数填充
   - 众数填充
   - 回归填充
```python
# 均值填充
df.fillna(df.mean())
中位数填充
df.fillna(df.median())
前向/后向填充
df.fillna(method='ffill') # 前向填充
df.fillna(method='bfill') # 后向填充
```
### 3. 高级填充方法
- **回归填充**：使用其他变量预测缺失值
- **KNN填充**：基于相似样本进行填充
- **多重插补**：考虑不确定性的填充方法

## 实践案例

### 数据探索
```python
查看缺失值情况
def missing_values_analysis(df):
missing = df.isnull().sum()
missing_percent = missing / len(df) 100
return pd.DataFrame({
'Missing Values': missing,
'Percentage': missing_percent
}).sort_values('Percentage', ascending=False)
```


### 处理流程
1. 分析缺失值模式
2. 选择合适的处理方法
3. 实施缺失值处理
4. 评估处理效果

### 注意事项
- 了解数据缺失的原因
- 评估不同处理方法的影响
- 保留原始数据的副本
- 记录处理过程和依据