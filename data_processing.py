import pandas as pd
import glob
def extract_number(file_name):
    # 提取文件名中的数字部分，例如 "citeseer_32" 提取出 "32"
    return int(''.join(filter(str.isdigit, file_name)))
folder_path = "D:/code/LLM/LLMTTT/all_data/concept_degree/density"
# 获取所有Excel文件的文件名
# for key in ['cora', 'pubmed', 'citeseer', 'wikics', 'arxiv']:
# for key in ['cora', 'pubmed', 'citeseer', 'wikics']:
for key in ['density']:
    # key = 'rim'
    filename_key = f"{folder_path}/*_{key}_*.xlsx"
    # filename_key = f"{folder_path}/{key}_*.xlsx"
    # file_names = glob.glob(folder_path + f'{key}_*.xlsx')  # 根据实际文件名的规律进行修改，例如data*.xlsx表示以"data"开头，以".xlsx"结尾的文件名
    file_names = glob.glob(filename_key)
    # 创建空的 DataFrame 用于存储结果
    # result = pd.DataFrame(columns=['File Name', 'value', 'params_changed_1', 'params_changed_2', 'params_lr_ttt_1', "params_lr_ttt_2"])
    # result = pd.DataFrame(columns=['File Name', 'value', 'params_changed', 'params_lr_ttt'])
    result = pd.DataFrame(columns=['File Name', 'value', 'params_lr_ttt_1', "params_lr_ttt_2"])
    # result = pd.DataFrame(columns=['File Name', 'value', "params_lr_ttt_2"])
    # result = pd.DataFrame(columns=['File Name', 'value', 'params_lr', 'params_num_layers', 'params_weight_decay', "params_dropout"])
    # 循环读取并处理每个文件
    for file_name in file_names:
        # 读取 Excel 文件
        df = pd.read_excel(file_name, engine='openpyxl')
        max_row = df.loc[df['value'].idxmax()]
        # 获取文件名，并作为列名
        file_name = file_name.replace('_.xlsx', '')  # 去掉文件扩展名
        file_name = file_name.replace(folder_path, '')
        max_row['File Name'] = file_name
        # result = result._append(max_row[['File Name', 'value', 'params_changed', 'params_lr_ttt']], ignore_index=True) # 将每个文件的最大值添加到结果 DataFrame 中
        # result = result._append(max_row[['File Name', 'value', 'params_changed_1', 'params_changed_2', 'params_lr_ttt_1', "params_lr_ttt_2"]], ignore_index=True)  # 将每个文件的最大值添加到结果 DataFrame 中
        result = result._append(max_row[['File Name', 'value', 'params_lr_ttt_1', "params_lr_ttt_2"]], ignore_index=True)  # 将每个文件的最大值添加到结果 DataFrame 中
        # result = result._append(max_row[['File Name', 'value', "params_lr_ttt_2"]], ignore_index=True)
        # result = result._append(max_row[['File Name', 'value', 'params_lr', 'params_num_layers', 'params_weight_decay', "params_dropout"]], ignore_index=True)  # 将每个文件的最大值添加到结果 DataFrame 中
    # 根据文件名中的数字进行排序
    # result['Sort Key'] = result['File Name'].apply(extract_number)
    # result = result.sort_values(by='Sort Key')
    # # 删除排序关键字列
    # result.drop('Sort Key', axis=1, inplace=True)
    output = f"{folder_path}/summarize.xlsx"
    # 保存结果 DataFrame 到新的 Excel 文件
    with pd.ExcelWriter(output, mode='a') as writer:
        result.to_excel(writer, sheet_name=key, index=False) #sheet_name 可以指定为本次调参目的

# db = "cora"
# b = 83
# filename = f"D:/code/LLM/LLMTTT/some TTT results/{db}_{b}_pagerank_with_gcn_none.csv"
# df = pd.read_csv(filename)
# df.to_excel(f"D:/code/LLM/LLMTTT/{db}_{b}_pagerank_with_gcn_none.xlsx", index=False)