import json

with open('bart_summary_evaluation.jsonl', 'r') as json_file:
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
print(f"      Basic errors: {sum(n_basic_err)}")
print(f"Commonsense errors: {sum(n_comm_err)}")

import pdb;pdb.set_trace()
