# K-평균 알고리즘

## K-Mean?
비지도학습

군집화 문제에 사용. k개의 군집으로 그롭화할 때 사용.

중심(centroid)라고 하는 k개의 점으로 계산, 데이터는 k개의 군집 중 하나에만 속할 수 있음.
한 군집 내의 모든 데이터들은 다른 centroid보다 자기 군집 centroid에 (거리가) 가까움

직접 오차함수 최적화 -> NP Hardness Problem
따라서 휴리스틱한 방법을 이용 (e.g. 반복개선)

## 반복 개선
세개의 단계로 구성
1. 초기단계: k개 중심의 초기 집합 설정
1. 할당단계: 각 데이터를 가장 가까운 군집에 할당
1. 업데이트 단계: 각 그룹에 대해 새로운 중심 설정

초기단계의 데이터는 데이터 중 k개의 데이터를 임의로 선택하여 설정

그 이후에는 각 군집의 데이터를 이용하여 중심 계산하여 2~3단계 반복


## Code

```
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.33:
        vectors_set.append([
            np.random.normal(1.0, 0.9),
            np.random.normal(3.0, 0.9)
        ])
    elif np.random.random() > 0.66:
        vectors_set.append([
            np.random.normal(3.0, 0.9),
            np.random.normal(2.0, 0.9)
        ])
    else:
        vectors_set.append([
            np.random.normal(5.0, 0.9),
            np.random.normal(5.0, 0.9)
        ])
    
df = pd.DataFrame({
    'x': [v[0] for v in vectors_set],
    'y': [v[1] for v in vectors_set],
})
sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
plt.savefig('./k-mean/data.jpg')

# -- data generation end --

vectors = tf.constant(vectors_set)
k = 5
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k, -1]))  # 중심점 임의로 설정

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)

means = tf.concat([tf.reduce_mean(tf.gather(vectors,
    tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
    reduction_indices=[1]) for c in range(k)], 0)

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(100):  # Number of Epoch
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

    data = {
        'x': [],
        'y': [],
        'cluster': [],
    }

    for i in range(len(assignment_values)):
        data['x'].append(vectors_set[i][0])
        data['y'].append(vectors_set[i][1])
        data['cluster'].append(assignment_values[i])

    df = pd.DataFrame(data)
    sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='cluster', legend=False)
    sns.plt.suptitle('epoch='+str(step))
    plt.savefig('./k-mean/'+str(step)+'.jpg')
    plt.close()
```

![Data spot graph](http://i.imgur.com/Qg3UTMx.jpg)
![Group Changing while epoch increase](http://i.imgur.com/YUM5GWL.gif)

*epoche==20 정도부터 큰 변화가 없다*


### 데이터의 군집 계산
```
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)
```
`vectors`의 차원은 `TensorShape([Dimension(2000), Dimension(2)])`이고 `centroides`의 차원은 `TensorShape([Dimension(4), Dimension(2)])`

`vectors`와 `centroides`의 차원이 다르기 때문에 직접 연산이 불가능하므로 `expand_dims` 메소드를 이용해서 차원을 변경해줌

Dimension of `vectors` is `TensorShape([Dimension(1), Dimension(2000), Dimension(2)])`

Dimension of `centroides` is `TensorShape([Dimension(4), Dimension(1), Dimension(2)])`

** 1차원 텐서는 차원이 다른 텐서와 연산 시 해당 차원에 맞게 계산을 반복

*tf.subtract()는 이전 버전 tf에서 tf.sub()로 사용되었음*

assignments 연산 중 `tf.reduce_sum()` 메소드는 지정한 차원을 따라 원소들을 더하는 역할
`D2`에는 x, y 좌표가 저장되어있었고, 해당 메소드를 실행한 후 텐서 차원이
`TensorShape([Dimension(4), Dimension(2000), Dimension(2)])` 에서
`TensorShape([Dimension(4), Dimension(2000)])`로 바뀜

### 새로운 Centroid 설정
```
means = tf.concat([tf.reduce_mean(tf.gather(vectors,
    tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
    reduction_indices=[1]) for c in range(k)], 0)

update_centroides = tf.assign(centroides, means)
```

*tf.concat() 메소드는 이전 버전과 비교해서 list와 int의 순서가 바뀌었음*