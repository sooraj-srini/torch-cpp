import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import io
def save_tensor(my_tensor, name):
    # my_tensor = torch.rand(3, 3).to(torch.device('cpu'));
    # print("[python] my_tensor: ", my_tensor)
    f = io.BytesIO()
    torch.save(my_tensor, f, _use_new_zipfile_serialization=True)
    with open(name, "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())

SUITE_ID = 337
benchmark_suite = openml.study.get_suite(SUITE_ID)  

if __name__ == '__main__':
    count = 5
    def get_task_size(id):
        task = openml.tasks.get_task(id)
        return task.get_X_and_y()[0].shape[1]
    arr = benchmark_suite.tasks.copy()
    arr.sort(key=lambda x: get_task_size(x))
    for task_id in arr:
        task = openml.tasks.get_task(task_id)
        count += 1
        print("Current task: ", count)
        print(task)
        dataset = task.get_dataset()
        print(f"Current Dataset:{dataset.name}")
        X, y, _, _ = dataset.get_data(target=task.target_name)
        np.random.seed(42)
        rng = np.random.permutation(X.shape[0])
        scaler = StandardScaler()

        labels = y.to_numpy()
        classes = np.unique(labels)
        labels = np.where(labels == classes[0], 0, 1)
        data_x = scaler.fit_transform(X)
        data_x, labels = data_x[rng], labels[rng]

        d_tensor = torch.tensor(data_x)
        d_labels = torch.tensor(labels)
        save_tensor(d_tensor, dataset.name + "_features.pt")
        save_tensor(d_labels, dataset.name + "_labels.pt")
