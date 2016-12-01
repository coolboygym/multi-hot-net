import numpy as np
import tensorflow as tf
import time
import pickle
from scipy.sparse import csr_matrix

batch_size = 100
field_numbers = 17
training_file = "../data/train0.embed.shuffle"

# get the size of each field (for example, size of weekday is 7)
def get_field_size(filename, field_number, full_data):
    with open(filename) as fin:
        arr = np.zeros((field_number,), dtype=np.int)  # size of each field
        for line in fin:
            nums = line.strip().split()
            if full_data:
                for i in range(len(nums) - 1):  # the last field is multi-hot
                    if int(nums[i]) > arr[i]:
                        arr[i] = int(nums[i])
                if len(nums) > 29:
                    multi_hot = nums[29]
                    temp = multi_hot.strip().split(',')
                    for i in range(len(temp)):
                        if int(temp[i]) > arr[29]:
                            arr[29] = temp[i]
            else:
                for i in range(len(nums)):
                    if int(nums[i]) > arr[i]:
                        arr[i] = int(nums[i])
    total_size = 0
    for i in range(len(arr)):
        arr[i] += 1  # from index number to field size
        total_size += arr[i]

    shift_size_arr = np.array(range(len(arr)))
    shift_size_arr[0] = 0
    tmp = 0
    for i in range(1, len(arr)):
        tmp += arr[i - 1]
        shift_size_arr[i] = tmp
    return total_size, arr, shift_size_arr

def read_data(filename, field_number):
    line_number = 0
    with open(filename) as fin:
        for _ in fin:
            line_number += 1
    print "##### there are ", line_number, "items in the training file #####"

    data_cols = np.array([range(field_number) for _ in range(line_number)])  # read data from data file
    line_number = 0
    with open(filename) as fin:
        for line in fin:
            nums = line.strip().split('\t')
            for i in range(len(nums)):
                data_cols[line_number][i] = int(nums[i])
            line_number += 1
    return line_number, data_cols

cols_size, size_arr, shift_arr = get_field_size(training_file, field_number=field_numbers, full_data=False)
item_numbers, cols = read_data(training_file, field_number=field_numbers)
# rows = np.vstack([range(item_numbers) for _ in range(field_numbers)]).transpose()
cols += shift_arr
data = np.ones_like(cols)

def get_batch_data(batch_id):
    global cols, data, batch_size
    batch_rows = np.vstack([range(batch_size) for _ in range(field_numbers)]).transpose()
    data_batch = csr_matrix((data.flatten()[batch_id*batch_size*field_numbers:(batch_id+1)*batch_size*field_numbers],
                             (batch_rows.flatten(),
                              cols.flatten()[batch_id*batch_size*field_numbers:(batch_id+1)*batch_size*field_numbers])),
                            shape=[batch_size, cols_size]).toarray()
    return data_batch

# ttt = get_batch_data(0)
# print ttt
# tt = get_batch_data(1)
# print tt
# for hj in range(cols_size):
#     if ttt[0][hj] == 1:
#         print hj,
# print '\n'
# for hj in range(cols_size):
#     if tt[0][hj] == 1:
#         print hj,
# exit(0)

# data_full = csr_matrix((data.flatten(), (rows.flatten(), cols.flatten())), shape=[item_numbers, cols_size]).toarray()

# print rows
# print cols
# print data_full
# exit(0)

graph = tf.Graph()

# embedding size of each field
# temp_k = [10 for _ in range(field_numbers)]
temp_k = [5, 5, 5, 5, 10, 10, 10, 5, 10, 5, 40, 5, 5, 5, 5, 10, 5]
k = np.array(temp_k)

# size of each field's hidden layer
hidden_size = np.array(range(field_numbers))
for ii in range(field_numbers):
    temp_size = 0
    for jj in range(field_numbers):
        if jj != ii:
            temp_size += k[jj]
    hidden_size[ii] = temp_size

# index = [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]]
indexes = np.array([range(field_numbers-1) for _ in range(field_numbers)])
for ii in range(field_numbers):
    temp_index = 0
    for jj in range(field_numbers):
        if jj != ii:
            indexes[ii][temp_index] = jj
            temp_index += 1

# print k
# print hidden_size
# print indexes
# exit(0)

# graph definition
with graph.as_default():
    embedding = [tf.Variable(tf.random_uniform([size_arr[i], k[i]], minval=-1, maxval=1)) for i in range(field_numbers)]
    inputs = [tf.placeholder(tf.float32, (batch_size, size_arr[i])) for i in range(field_numbers)]
    embed = [tf.nn.relu(tf.matmul(inputs[i], embedding[i])) for i in range(field_numbers)]

    w = [tf.Variable(tf.random_normal([hidden_size[i], size_arr[i]])) for i in range(field_numbers)]
    b = [tf.Variable(tf.zeros([size_arr[i]])) for i in range(field_numbers)]
    h = [tf.concat(1, [embed[indexes[i][j]] for j in range(field_numbers-1)]) for i in range(field_numbers)]
    x = [tf.matmul(h[i], w[i]) + b[i] for i in range(field_numbers)]
    l = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(x[i], inputs[i])) for i in range(field_numbers)]

    loss = tf.add_n([1.0 / field_numbers * l[i] for i in range(len(l))])
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.initialize_all_variables()

# train model
num_round = 300

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    for i in range(num_round):
        t_start = time.time()
        for j in range(item_numbers / batch_size - 300):  # use last 200 batch for valid
            batch_data = get_batch_data(j)
            feed_dict = {}
            for s in range(field_numbers):
                feed_dict[inputs[s]] = batch_data[:, shift_arr[s]:(shift_arr[s]+size_arr[s])]
            _, loss_train, embed_matrix = session.run([optimizer, loss, embedding], feed_dict=feed_dict)
        print "num_round = ", i, "\ttrain_loss = ", loss_train, "\titer_time = ", time.time() - t_start
        pickle.dump(embed_matrix, open("../log/embed_matrix_%d" % i, 'w'))

        t_start_val = time.time()
        for j in range(item_numbers / batch_size - 300, item_numbers / batch_size):
            batch_data = get_batch_data(j)
            feed_dict = {}
            for s in range(field_numbers):
                feed_dict[inputs[s]] = batch_data[:, shift_arr[s]:(shift_arr[s]+size_arr[s])]
            loss_val = session.run([loss], feed_dict=feed_dict)
        print "num_round = ", i, "\tvalid_loss = ", loss_val, "\titer_time = ", time.time() - t_start_val
