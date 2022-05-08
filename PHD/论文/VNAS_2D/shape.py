

dilation = 1

input = 64
padding = 1
kernel = 3
stride = 1

print((input + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
