from experiment.accuracy_experiment import *

master_path = './master_test_data.csv'
fd_path = './fd_test.txt'

df_master, fd_list = load_master(master_path, fd_path)
df_master = check_df(df_master, fd_list)

target_generation = generate_target(df_master, fd_path, error_num=2, percentage=0.9)
# 最好把原始的original value 给到 violate_cell, 这样我们就不用自己去比较这个错误到底是不是错误了, repair到底对否.
df_target = target_generation[0]
print(df_target)

violate_cells = calculate_error_facotors(fd_path, df_target)
violate_cells = divide_budget(budget=10, violate_cells=violate_cells)
violate_cells = find_repair_value(violate_cells, df_master, df_target, threshold=0.9)
violate_cells = rank_cell(violate_cells)

print('---------------------------------------------------------------------------------')
for c in violate_cells:
    row = c.row
    col = c.col
    value = c.value
    involved_errors = c.involved_errors  ##分子
    total_errors = c.total_errors  ##分母
    error_factor = c.error_factor  ##求值
    budget = c.budget
    repair_values = c.repair_values
    print('row', row, 'col', col, 'target_value', value, 'involved_errors', involved_errors, total_errors,
          'error_factor', error_factor, budget, 'repair_values', repair_values)
    print('+++++++++++++++++++++++++')
