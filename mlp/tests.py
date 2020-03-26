from utils import *

if __name__ == "__main__":
    batch = LoadBatch("data_batch_1")
    print(batch.keys())
    print(len(batch[b"labels"]))
    print(batch[b"data"].shape)

    for i in range(100):
        plot(batch[b"data"][i])
