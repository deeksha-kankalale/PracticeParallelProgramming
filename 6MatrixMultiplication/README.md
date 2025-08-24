<<<<<<< HEAD
Naive Matrix multiplication

One thread per C[y, x]   

Loop over k from 0 to SIZE–1, multiplying A[y,k] * B[k,x] and summing.   

No optimizations — no tiling, no shared memory, no register blocking, no loop splitting   
=======
The dot-product is update in the kernel
C[m*WIDTH + n] += A[m*WIDTH + k] * B[k*WIDTH + n]; is correct for row-major.  

Setting one thread to compute one C[m,n] is the standard naïve pattern.  

Using a single block of (WIDTH, WIDTH) for a WIDTH×WIDTH matrix is okay for this fixed-size toy (though not scalable).  
>>>>>>> a991e0e (Added naive multiplication)
