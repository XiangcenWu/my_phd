# external packages
import torch

# self defined packages
import operations as op



def main():
    reduce=[
        ('max_pool_3x3', 0), 
        ('max_pool_3x3', 1), 
        ('skip_connect', 2), 
        ('max_pool_3x3', 1), 
        ('max_pool_3x3', 0), 
        ('skip_connect', 2), 
        ('skip_connect', 2), 
        ('max_pool_3x3', 1)
    ]
    x, y = zip(*reduce)
    print(x)
    print(y)

    z = zip(*reduce)
    for i in z:
        print(i)


    c = 432455435342
    x = [int(i) for i in str(c)]
    print(x)
    print(torch.hub.get_dir())
    


if __name__ == '__main__':
    main()
