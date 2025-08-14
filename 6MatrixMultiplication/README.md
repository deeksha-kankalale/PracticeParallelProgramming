Naive Matrix multiplication

One thread per C[y, x]   

Loop over k from 0 to SIZE–1, multiplying A[y,k] * B[k,x] and summing.   

No optimizations — no tiling, no shared memory, no register blocking, no loop splitting   
