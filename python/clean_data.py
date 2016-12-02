import numpy as np
import pickle
import random
import linecache

def shuffle_data(filename, result_file):
    with open(filename) as fin:
        with open(result_file, 'w') as fout:
            item_number = 0
            for _ in fin:
                item_number += 1
            temp_x = range(item_number)
            for _ in range(7):
                random.shuffle(temp_x)
            for i in temp_x:
                line = linecache.getline(filename, i+1)
                fout.write(line)

def negative_down_sampling(filename, result_file, sample_rate):
    with open(filename) as fin:
        with open(result_file, 'w') as fout:
            for line in fin:
                nums = line.strip().split()
                if int(nums[0]) == 0:
                    temp_p = np.random.random()
                    if temp_p < sample_rate:
                        fout.write(line)
                else:
                    fout.write(line)

# delete field which is no use for embed (the first field is click, do not use)
def delete_no_use_field(filename, result_file, delete_list, remain_label):
    no_use_field_index = np.array(delete_list)
    with open(filename) as fin:
        with open(result_file, 'w') as fout:
            if remain_label:
                start_index = 0
            else:
                start_index = 1

            for line in fin:
                nums = line.strip().split('\t')
                for i in range(start_index, 30):
                    if i not in no_use_field_index:
                         fout.write(nums[i] + '\t')
                fout.write('\n')

# get the size of each field (for example, size of weekday is 7)
def get_field_size(filename, field_number, full_data):
    with open(filename) as fin:
        arr = np.zeros((field_number,), dtype=np.int)  # size of each field
        for line in fin:
            nums = line.strip().split('\t')
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
        arr[i] += 1
        total_size += arr[i]

    shift_size_arr = np.array(range(len(arr)))
    shift_size_arr[0] = 0
    tmp = 0
    for i in range(1, len(arr)):
        tmp += arr[i - 1]
        shift_size_arr[i] = tmp
    return total_size, arr, shift_size_arr

def clean_embed_test_file(filename):
    with open(filename) as fin:
        with open(filename + '.clean', 'w') as fout:
            for line in fin:
                nums = line.strip().split()
                if not(int(nums[5]) == 218 or int(nums[11]) >= 8269 or int(nums[16]) == 132):
                    fout.write(line)

# change the state of file to generate training file for fnn
def change_file_to_train(filename, result_file, delete_list, shift_array):
    no_use_field_index = np.array(delete_list)
    with open(filename) as fin:
        with open(result_file, 'w') as fout:
            for line in fin:
                shift_index = 0
                nums = line.strip().split()
                fout.write(nums[0] + ' ')
                for i in range(1, 30):
                    if i not in no_use_field_index:
                        fout.write(str(shift_array[shift_index] + int(nums[i])) + ':1 ')
                        shift_index += 1
                fout.write('\n')

def generate_model_file(embed_file, result_file, total_size, name_field):
    with open(result_file, 'w') as fout:
        line_index = 0
        embed_matrix = pickle.load(open(embed_file))
        embed_size_array = []   # embed size of each field
        for i in range(len(embed_matrix)):
            embed_size_array.append(len(embed_matrix[i][0]))

        temp_weight = np.random.random([1,1])[0][0] * 2 - 1
        fout.write(str(temp_weight) + ' ' + str(total_size) + ' ')
        for i in range(len(embed_size_array)):
            fout.write(str(embed_size_array[i]) + ' ')
        fout.write('\n')
        for i in range(len(embed_matrix)):  # field number
            for j in range(len(embed_matrix[i])):  # size of each field
                fout.write(str(line_index) + ' ')
                line_index += 1
                fout.write(str(embed_size_array[i]) + ' ')
                for k in range(embed_size_array[i]):  # embed size of each field
                    fout.write(str(embed_matrix[i][j][k]) + ' ')
                fout.write(name_field[i] + ':' + str(j) + '\n')

def generate_test_file(filename, result_file, shift_array):
    with open(filename) as fin:
        with open(result_file, 'w') as fout:
            for line in fin:
                nums = line.strip().split()
                fout.write(nums[0] + ' ')
                for i in range(1, len(nums)):
                    fout.write(str(int(nums[i])+shift_array[i-1]) + ':' + '1' + ' ')
                fout.write('\n')


if __name__ == '__main__':
    field_to_delete = [3, 4, 5, 6, 16, 17, 18, 25, 26, 27, 28, 29]
    inverse_dict = {0:'weekday', 1:'hour', 2:'user_agent_os', 3:'user_agent_browser', 4:'IP0', 5:'IP1', 6:'IP2',
                   7:'region', 8:'city', 9:'adexchange', 10:'domain', 11:'slotwidth', 12:'slotheight',
                   13:'slotvisibility', 14:'slotformat', 15:'slotprice', 16:'creative'}

    name_fields = {'weekday':0, 'hour':1, 'user_agent_os':2, 'user_agent_browser':3, 'IP0':4, 'IP1':5, 'IP2':6,
                   'region':7, 'city':8, 'adexchange':9, 'domain':10, 'slotwidth':11, 'slotheight':12,
                   'slotvisibility':13, 'slotformat':14, 'slotprice':15, 'creative':16}
    # shuffle_data('../data/train0.embed', '../data/train0.embed.shuffle')
    # exit(0)
    # file_name = '../data/train0.embed'
    # l = linecache.getline(file_name, 248546)
    # delete_no_use_field(filename="../data/test0", result_file="../data/test0.embed",
    #                     delete_list=field_to_delete, remain_label=True)
    # exit(0)
    # clean_embed_test_file("../data/test0.embed")
    # exit(0)
    # negative_down_sampling('../data/test0.embed.clean', '../data/test0.embed.clean.sample', 0.3)
    # exit(0)
    total_sizes, size_arr, shift_arr = get_field_size("../data/train0.embed", field_number=17, full_data=False)
    print total_sizes, '\n', size_arr, '\n', shift_arr
    # exit(0)
    # change_file_to_train("../data/train0", "../data/train0.fnn.train", field_to_delete, shift_array=lll)

    generate_model_file(embed_file='../log/embed_matrix_160', result_file='../data/train0.fnn.model.160',
                        total_size=total_sizes, name_field=inverse_dict)
    # generate_test_file('../data/test0.embed.clean.nds', '../data/test0.fnn.test', shift_arr)
