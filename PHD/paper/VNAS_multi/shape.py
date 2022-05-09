

dilation = 1

input = 64
padding = 2
kernel = 5
stride = 2

print((input + 2*padding - dilation*(kernel-1) - 1)/stride + 1)


