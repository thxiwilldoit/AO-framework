import copy
import time

import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from torchvision import transforms
from multiprocessing import Pool, Manager, shared_memory
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import matplotlib.cm as cm

def make_index(num_job, num_mas, num_eve_ope, ope_ma_adj_batch, sequence_batch, fig_size=40):  # num_eve_ope==nums_ope_batch
    # num_job: number of jobs
    # num_mas: number of machines
    # num_eve_ope: the tensor that contains the number of operations per jobs
    # ope_ma_adj_batch: the tensor that can process 0/1 machines per operation
    # sequence_batch is all -1 by default, and the length is the number of operations
    # fig_size is the size of the generated picture
    batch_num = len(num_eve_ope)
    con_o_o = []
    mas_index_y = np.linspace(0, fig_size - 1, num_mas + 2)[1:-1]
    index_ma = [[elem, fig_size - 2] for elem in mas_index_y]
    index_ma = torch.tensor(index_ma)
    con_m_o = ope_ma_adj_batch.clone()
    con_m_o = con_m_o.view(batch_num, -1)
    step_num = (sequence_batch != -1).sum(dim=1)
    index_ope_batch = []
    ope_index_y = np.linspace(0, fig_size - 1, num_job + 2)[1:-1]
    for ope_n in range(len(num_eve_ope[0])):
        ope_index_x = np.linspace(0, fig_size - 1, num_eve_ope[0][ope_n] + 2)[1:-1]
        combined = [[elem, ope_index_y[ope_n]] for elem in ope_index_x]
        index_ope_batch = index_ope_batch + combined
    index_ope = torch.tensor(index_ope_batch)

    for i_batch in range(batch_num):
        # Generate operation and connection matrix between operations
        sequence = sequence_batch[i_batch].clone()
        adjacency_matrix = torch.zeros((sum(num_eve_ope[i_batch]), sum(num_eve_ope[i_batch])), dtype=torch.int)
        for i in range(len(sequence) - 1):
            if sequence[i] == -1 or sequence[i + 1] == -1:
                break
            adjacency_matrix[sequence[i], sequence[i + 1]] = 1
        adjacency_matrix = adjacency_matrix.view(-1)
        con_o_o.append(adjacency_matrix)

    con_o_o = torch.stack(con_o_o)
    # Coordinates of operation/
    # coordinates of machine/
    # adjacency matrix between operation and machine/
    # adjacency matrix between operations/
    # which step is currently running
    return index_ope, index_ma, con_m_o, con_o_o, step_num

def one_batch_generate_plot(i_batch, operations, machines, connections_m_o, connections_o_o,
                            step_num_batch, a_count, b_count, save_path, transform, shm_name, flag_train, proc_time_batch):
    x, y = zip(*operations)
    m, n = zip(*machines)
    min_val = proc_time_batch.min()
    max_val = proc_time_batch.max()
    bn_proc_time = np.copy(((proc_time_batch - min_val) / (max_val - min_val)).reshape(-1))
    colors = cm.viridis(bn_proc_time)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=50, facecolor='white')
    ax.scatter(x, y, color='black', alpha=1, s=10)
    ax.scatter(m, n, color='black', alpha=1, s=5)
    if flag_train:
        save_path_batch = save_path + "train/" + str(i_batch) + '-batch/'
    else:
        save_path_batch = save_path + "evaluate/" + str(i_batch) + '-batch/'
    if not os.path.exists(save_path_batch):
        os.makedirs(save_path_batch)

    for i in range(a_count):
        for j in range(b_count):
            if connections_m_o[i * b_count + j] == 1:
                ax.plot([x[i], m[j]], [y[i], n[j]], color=colors[i * b_count + j], linewidth=0.5)

    for i in range(a_count):
        for j in range(a_count):
            if connections_o_o[i * a_count + j] == 1:
                ax.plot([x[i], x[j]], [y[i], y[j]], color='black', linewidth=0.5)

    ax.axis('off')
    file_name = "step" + str(int(step_num_batch)) + ".png"
    fig.savefig(save_path_batch + file_name, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)
    image = Image.open(save_path_batch + file_name).convert('RGB')
    image_tensor = transform(image)
    shm = shared_memory.SharedMemory(name=shm_name)

    for i_channel in range(3):
        start = i_batch * image_tensor.numel() * 4 + i_channel * image_tensor[i_channel].numel() * 4
        end = i_batch * image_tensor.numel() * 4 + (i_channel + 1) * image_tensor[i_channel].numel() * 4
        shm.buf[start:end] = image_tensor[i_channel].numpy().tobytes()

def generate_scatter_plot(full_operations, full_machines, full_connections_m_o,
                          full_connections_o_o, step_num, flag_train, proc_time, fig_size=40):
    # Proc_time is the processing time of each operation
    matplotlib.use('Agg')
    # start_all_time = time.time()
    transform = transforms.Compose([
        transforms.Resize((fig_size, fig_size)),
        transforms.ToTensor(),
    ])
    save_path = './vis_scheduling/'
    batch_num = len(full_connections_m_o)
    img_list = []
    operations = full_operations.cpu().numpy()
    machines = full_machines.cpu().numpy()
    connections_m_o = full_connections_m_o.cpu().numpy()
    connections_o_o = full_connections_o_o.cpu().numpy()
    proc_time_m_o = proc_time.cpu().numpy()
    step_num = step_num.cpu().numpy()
    a_count = len(operations)
    b_count = len(machines)
    image_shape = (3, fig_size, fig_size)
    shm = shared_memory.SharedMemory(create=True, size=4 * batch_num * np.prod(image_shape))
    for i_batch in range(batch_num):
        one_batch_generate_plot(i_batch, operations, machines, connections_m_o[i_batch], connections_o_o[i_batch], step_num[i_batch],
        a_count, b_count, save_path, transform, shm.name, flag_train, proc_time_m_o[i_batch])
    for i_batch in range(batch_num):
        img_tensor = np.frombuffer(shm.buf[4 * i_batch * np.prod(image_shape): 4 * (i_batch + 1) * np.prod(image_shape)],
                                   dtype=np.float32).copy()
        img_tensor_buffer = torch.from_numpy(img_tensor).reshape(image_shape)
        img_list.append(img_tensor_buffer)
    img = torch.stack(img_list)
    shm.close()
    shm.unlink()
    return img

if __name__ == '__main__':
    # instances
    num_m = 3
    num_j = 3
    pro_time = torch.tensor([[[ 1.,  2.,  0.],
         [ 0.,  3.,  4.],
         [ 5.,  0.,  6.],
         [ 7.,  8.,  0.],
         [ 0.,  9., 10.],
         [11., 12.,  0.],
         [13.,  0., 14.]],
        [[ 1.,  2.,  0.],
         [ 0.,  3.,  4.],
         [ 5.,  0.,  6.],
         [ 7.,  8.,  0.],
         [ 0.,  9., 10.],
         [11., 12.,  0.],
         [13.,  0., 14.]],
        [[ 1.,  2.,  0.],
         [ 0.,  3.,  4.],
         [ 5.,  0.,  6.],
         [ 7.,  8.,  0.],
         [ 0.,  9., 10.],
         [11., 12.,  0.],
         [13.,  0., 14.]],
        [[ 1.,  2.,  0.],
         [ 0.,  3.,  4.],
         [ 5.,  0.,  6.],
         [ 7.,  8.,  0.],
         [ 0.,  9., 10.],
         [11., 12.,  0.],
         [13.,  0., 14.]],
        [[ 1.,  2.,  0.],
         [ 0.,  3.,  4.],
         [ 5.,  0.,  6.],
         [ 7.,  8.,  0.],
         [ 0.,  9., 10.],
         [11., 12.,  0.],
         [13.,  0., 14.]]])
    ope_l = torch.tensor([[2, 2, 3],
        [2, 2, 3],
        [2, 2, 3],
        [2, 2, 3],
        [2, 2, 3]])
    ope_ma_adj_batch = torch.tensor([[[1, 1, 0],
         [0, 1, 1],
         [1, 0, 0],
         [1, 1, 0],
         [0, 1, 1],
         [1, 1, 0],
         [1, 0, 1]],
        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 1]],
        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 1, 0],
         [1, 0, 1]],
        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 1]],
        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 0],
         [0, 0, 1],
         [1, 1, 0],
         [1, 0, 1]]])
    sequence_batch = torch.tensor([
        [ 0,  2,  4,  3, -1, -1, -1],
        [ 0,  2,  4,  1, -1, -1, -1],
        [ 0,  2,  4,  1, -1, -1, -1],
        [ 4,  0,  2,  1, -1, -1, -1],
        [ 2,  0,  4,  1, -1, -1, -1]])
    fig_size = 48
    start_time = time.perf_counter()
    list_k = make_index(num_j, num_m, ope_l, ope_ma_adj_batch, sequence_batch, fig_size)
    z = generate_scatter_plot(list_k[0], list_k[1], list_k[2], list_k[3], list_k[4], True, pro_time, fig_size)
    end_time = time.perf_counter()
    print(f"Run time: {end_time - start_time:.6f} s")
