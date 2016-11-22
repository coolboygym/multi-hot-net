import numpy as np


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
    return total_size, arr


# delete field which is no use(the first field is click, do not use)
def delete_no_use_field(filename):
    no_use_field_index = np.array([0, 3, 5, 6, 9, 10, 11, 15, 16, 17, 18, 25, 27, 28, 29])
    with open(filename) as fin:
        with open(filename + '_clean', 'w') as fout:
            for line in fin:
                nums = line.strip().split('\t')
                for i in range(30):
                    if i not in no_use_field_index:
                         fout.write(nums[i] + '\t')
                fout.write('\n')


if __name__ == '__main__':
    ll, lll = get_field_size("../data/train0_clean", field_number=15, full_data=False)
    print lll
    # delete_no_use_field("../data/train0")
