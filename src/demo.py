from src import algo


def nlof_mll_demo():
    # Simple demo with only 2 epoch
    # Simple topology:  H1 ---- S1 ---- H2
    flows_batch = [
        [[20, 20.5, 21, 19, 12],
         [20, 20.5]],
        [[19, 20.5, 18, 20, 12],
         [19, 20.5]]
    ]
    paths_batch = [None, None]
    paths_batch[0] = [
        [[1, 1], [1, 1], [1, 0], [1, 0], [1, 0]],
        [[1, 1], [1, 1]]
    ]
    paths_batch[1] = paths_batch[0]

    wages_batch = algo.learn_nlof_mll(2, flows_batch, paths_batch)
    print(wages_batch)


if __name__ == '__main__':
    nlof_mll_demo()
