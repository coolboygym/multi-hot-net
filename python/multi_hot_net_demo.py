import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix


def latent_func(arr, shift_val=1):
    arr += shift_val
    arr %= 20
    # for i in range(arr.size):
    #     arr[i] = (arr[i] + 1) % 20
# exit(0)

rows = np.vstack([range(10000) for i in range(5)]).transpose()
cols = np.random.randint(0, 20, [10000, 1])
cols = np.hstack([cols for i in range(5)])
for i in range(5):
    latent_func(cols[:, i], i)
cols += np.array([0, 20, 40, 60, 80])
data = np.ones_like(cols)

data_full = csr_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=[10000, 100]).toarray()

print rows
print cols
print data
print data_full
exit(0)

graph = tf.Graph()

k = 5  # embedding size
batch_size = 100

index = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]

# graph definition
with graph.as_default():
    embedding = [tf.Variable(tf.random_normal([20, k])) for i in range(5)]
    inputs = [tf.placeholder(tf.float32, (batch_size, 20)) for i in range(5)]
    embed = [tf.nn.relu(tf.matmul(inputs[i], embedding[i])) for i in range(5)]

    w = [tf.Variable(tf.random_normal([k * 4, 20])) for i in range(5)]
    b = [tf.Variable(tf.zeros([20])) for i in range(5)]
    h = [tf.concat(1, [embed[index[i][j]] for j in range(4)]) for i in range(5)]
    x = [tf.matmul(h[i], w[i]) + b[i] for i in range(5)]
    l = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x[i], inputs[i])) for i in range(5)]

    a0 = 0.2
    a1 = 0.2
    a2 = 0.2
    a3 = 0.2
    a4 = 0.2
    b1 = 0.2
    loss = (a0 * l[0] + a1 * l[1] + a2 * l[2] + a3 * l[3] + a4 * l[4]) #+ b1 * (tf.square(a0) + tf.square(a1) + tf.square(a2) + tf.square(a3) + tf.square(a4))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()

# train model
num_round = 1000
batch_size = 100

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for i in range(num_round):
        for j in range(10000 / batch_size):
            batch_data = data_full[j * batch_size:(j + 1) * batch_size]
            feed_dict = {}
            for k in range(5):
                feed_dict[inputs[k]] = batch_data[:, 20 * k:20 * (k + 1)]
            _, loss_val= session.run([optimizer, loss], feed_dict=feed_dict)
            print loss_val
            average_loss = average_loss + loss_val

