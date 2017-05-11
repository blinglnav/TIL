# 선형회귀분석

## 테스트 데이터 생성
```
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vector_set = []

for i in range(num_points):
    x = np.random.normal(0, 0.55)
    y = x * 0.1 + 0.3 + np.random.normal(0, 0.03)
    vector_set.append([x, y])

x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

plt.plot(x_data, y_data, 'ro')
plt.show()
```

![created data graph](http://i.imgur.com/779hUWi.png)

y = W * x + b

W = 0.1

b = 0.3

## 비용함수
x, y 데이터와 함수의 형태(y = W * x + b)가 주어졌을 때 W와 b를 알 수 있는 방법은 W와 b를 수정해가면서 오차가 가장 적도록 설정하는 것

이 때, W와 b를 통해 오차를 계산하는 함수 -> 비용함수(Cost function) 또는 오차함수(Error function)

이 경우 주어진 x 값들의 집합에 생성된 W와 b를 대입해 얻은 y 값들과 기존 y값들의 거리를 계산하기 위해 유클리드 거리를 이용

`dist^2 = (x1-x2)^2 + (y1-y2)^2`


## 경사하강법
Gradient Descent Optimization

오차함수의 크가가 최소화 되는 방향으로 매개변수들의 값을 수정하는 것을 반복하여 최종적으로 매개변수를 구하는 방법


## 코드
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def calc_error(x_data, y_data, W, b):
    count = 0
    error_sum = 0
    for x in x_data:
        error_sum += abs(y_data[count] - (W * x + b))
        count += 1
    return error_sum / count

if __name__ == '__main__':
    num_points = 1000
    vector_set = []

    for i in range(num_points):
        x = np.random.normal(0, 0.55)
        y = x * 0.1 + 0.3 + np.random.normal(0, 0.03)
        vector_set.append([x, y])

    x_data = [v[0] for v in vector_set]
    y_data = [v[1] for v in vector_set]

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.5)  # Training Rate
    #optimizer = tf.train.AdamOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    error_list = []

    for step in range(20):  # Training Epoch
        sW = sess.run(W)
        sb = sess.run(b)
        ce = calc_error(x_data, y_data, sW, sb)
        error_list.append(ce)
        print('W=' + str(sW) + 'b=' + str(sb) + 'mean error=' + str(ce))
        sess.run(train)

    sW = sess.run(W)
    sb = sess.run(b)
    ce = calc_error(x_data, y_data, sW, sb)
    error_list.append(ce)
    print('W=' + str(sW) + 'b=' + str(sb) + 'mean error=' + str(ce))

    plt.plot(range(len(error_list[1:])), error_list[1:])
    plt.show()
```

![Cost change graph while trainig](http://i.imgur.com/NteS0Xa.png)
![Cost changing state](http://i.imgur.com/GWej7dO.png)

### Training Rate
Training Rate 값이 크면 값을 찾기 전에 지나쳐버릴 수 있음. 반대로 너무 작으면 수렴하는데 시간이 오래 걸림

![Cost change graphe while trainig with rate=0.001](http://i.imgur.com/mVvyM3U.png)

왜인지는 모르겠지만 Rate를 매우 작은 값(0.001)로 설정할 경우 위와 같이 선형적으로 감소하면서 최종적으로 높은 cost를 가지는 그래프를 보이는데 아직 이유는 모르겠음

### Training Epoch
학습 횟수

횟수가 많아질 수록 더 정확한 값을 보여줌. but, 어느정도 이후에는 감소하는 정도가 매우 줄어드니 적절한 값을 찾아야함
