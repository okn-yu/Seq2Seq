import random

from sklearn.utils import shuffle


def to_ID():
    # 10: padding symbol
    # 11: minus
    # 12: start symbol
    _ = {str(i): i for i in range(10)}
    _.update({" ": 10, "-": 11, "_": 12})

    return _


def gen_num():
    # max 3 digit.
    num = [random.choice(list("0123456789")) for _ in range(random.randint(1, 3))]

    # this 'int' removes lead 0.
    # for ex, ['0' ,'2', '3'] -> 23
    return int("".join(num))


def add_padding(num, is_input=True):
    if is_input:
        # left justified, 7 digit.
        # input returns always 7 digit, as "123" "-" "456".
        return "{: <7}".format(num)
    else:
        # left justified, 5 digit.
        # output returns always 5 dig with start symbol, as "_" "-" "123".
        return "{: <5s}".format(num)


def subtraction_datasets(record_num):
    input_data = []
    output_data = []

    while len(input_data) < record_num:
        x = gen_num()
        y = gen_num()
        z = x - y
        input_char = add_padding(str(x) + "-" + str(y))
        output_char = add_padding("_" + str(z), is_input=False)

        char2id = to_ID()
        input_data.append([char2id[c] for c in input_char])
        output_data.append([char2id[c] for c in output_char])

    return input_data, output_data

def to_batch(input_date, output_data, batch_size=100):
    input_batch = []
    output_batch = []

    input_shuffle, output_shuffle = shuffle(input_date, output_data)

    for i in range(0, len(input_date), batch_size):
        input_batch.append(input_shuffle[i:i+batch_size])
        output_batch.append(output_shuffle[i:i+batch_size])

    return input_batch, output_batch