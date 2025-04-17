import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data as Data
from transformer_struct import train
from transformer_struct import test
from transformer_struct import Transformer


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_and_concat(file_list, features_num, src_len, tgt_len):
    """
    file_list: 多个 CSV 文件路径
    返回拼接后的 src 和 tgt 张量
    """
    all_src = []
    all_tgt = []

    for file in file_list:
        data = pd.read_csv(file).iloc[:, 1:].astype('float32')
        src_seq = []
        tgt_seq = []

        for i in range(len(data) - src_len - tgt_len):
            src_seq.append(data.iloc[i:i+src_len, 0:features_num].values)
            tgt_seq.append(data.iloc[i+src_len:i+src_len+tgt_len, -1].values.reshape(-1, 1))

        src_seq = torch.tensor(np.array(src_seq))
        tgt_seq = torch.tensor(np.array(tgt_seq))

        all_src.append(src_seq)
        all_tgt.append(tgt_seq)

    return torch.cat(all_src, dim=0), torch.cat(all_tgt, dim=0)


# ----------- 主程序 -----------  
if __name__ == '__main__':
    
    features_num = 6
    embedding_dim = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    max_len = 512
    dropout = 0.1

    Epoch = 10
    batch_size = 32
    lr = 0.0001
    src_len = 20  # 已知前20天的数据
    tgt_len = 10  # 预测后10天的数据
    # num_samples = 100  # 模拟训练集大小

    # 创建模型
    # model = Transformer(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)

    # torch.manual_seed(42)  # 设置随机种子，保证结果可复现

    # # 构造模拟数据集 [(src, tgt), ...]
    # train_data = []
    # for _ in range(num_samples):
    #     src = torch.rand(batch_size, src_len, features_num)
    #     tgt = torch.rand(batch_size, tgt_len, features_num)
    #     train_data.append((src, tgt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout).to(device)
    criterion = nn.MSELoss()  # 均方误差用于回归
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #加载数据
    # 训练集文件
    train_files = [
        'D:/Mjw/desktop/研究生学习/ML与DL/linear_regression_py/TRANSFORMER/ETTh1.csv',
        'D:/Mjw/desktop/研究生学习/ML与DL/linear_regression_py/TRANSFORMER/ETTh2.csv'
    ]

    # 测试集文件
    test_files = [
        'D:/Mjw/desktop/研究生学习/ML与DL/linear_regression_py/TRANSFORMER/ETTm1.csv',
        'D:/Mjw/desktop/研究生学习/ML与DL/linear_regression_py/TRANSFORMER/ETTm2.csv'
    ]

    # 调用函数
    train_src, train_tgt = process_and_concat(train_files, features_num, src_len, tgt_len)
    test_src, test_tgt = process_and_concat(test_files, features_num, src_len, tgt_len)
    print(train_src)
    print(train_src.shape)
    print(train_tgt.shape)

    train_dataset = Data.TensorDataset(train_src, train_tgt)
    test_dataset = Data.TensorDataset(test_src, test_tgt)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    #开始训练
    for epoch in range(1, Epoch + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        test(model, test_loader, criterion, device)

    # 训练模型
    # train_transformer(model, train_data, embedding_dim, num_epochs=5)