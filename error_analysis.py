import json
import numpy as np

models = ['bart', 't5', 'pegasus']

for model in models:
    filename = './data/annotated_' + model + '.jsonl'
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    # track num of each error by index
    n_basic_err = []
    n_comm_err = []

    for json_str in json_list:
        n_basic = 0
        n_comm = 0
        result = json.loads(json_str)
        if 'spans' in result:
            for span in result['spans']:
                n_basic += 1 if (span['label'] == 'BASIC') else 0
                n_comm += 1 if (span['label'] == 'COMMONSENSE') else 0

        n_basic_err.append(n_basic)
        n_comm_err.append(n_comm)

    total_err = [x+y for x,y in zip(n_basic_err, n_comm_err)]

    # statistics
    print(model + ' stats:')
    print('ERROR RATES')
    print(f"      Basic errors: {np.count_nonzero(n_basic_err)}%")
    print(f"Commonsense errors: {np.count_nonzero(n_comm_err)}%")
    print(f"      Total errors: {np.count_nonzero(total_err)}%")
    print("")
    print('ERROR TOTALS')
    print(f"      Basic errors: {sum(n_basic_err)}")
    print(f"Commonsense errors: {sum(n_comm_err)}")
    print(f"      Total errors: {sum(total_err)}")
    print('---')

    # import pdb;pdb.set_trace()
