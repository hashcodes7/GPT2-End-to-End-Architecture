They are **not random during inference**. They were **initialized randomly before training**, but during training they were updated millions/billions of times using gradient descent.
- `W_Q`: learns how to create **Queries** ("What am I looking for?")
- `W_K`: learns how to create **Keys** ("What information do I contain?")
- `W_V`: learns how to create **Values** ("What information should I pass on?")

w _q= [                             (this needs to be same size as x)
[ 0.5, -0.2],  
[ 0.1, 0.8],  
[-0.4, 0.3]  
]
w _k=[                               (this needs to be same size as x)
[-0.3, 0.6],  
[ 0.7, -0.1],  
[ 0.2, 0.5]  
]
w _v=[                                (this can be different size as x)
[ 0.2, -0.5, 0.1, 0.8],  
[ 0.6, 0.3, -0.4, 0.2],  
[-0.7, 0.4, 0.9, -0.1]  
]
# calculating QKV Vectors

1- **Q = X × W_Q** (5×2)
Q =  
[  
[ 0.193, -0.150],  
[-0.049, 0.665],  
[ 0.508, 0.126],  
[ 0.443, -0.707],  
[ 0.041, 0.672]  
]

2- K = X × WK (5x2)
K =  
[  
[-0.245, 0.637],  
[ 0.586,-0.047],  
[ 0.100,-0.045],  
[-0.775,0.999],  
[ 0.495,0.497]  
]

# 3 Value
W_V (3×2) =  
[  
[ 0.4, -0.3],  
[ 0.1, 0.8],  
[-0.5, 0.2]  
]
so -->
V =  X × W_V =
[  
[ 0.083, -0.260],  
[-0.055, 0.659],  
[ 0.517, 0.137],  
[ 0.209, -0.829],  
[-0.048, 0.562]  
]

# calculating attention scores (Scores = QK ᵀ )

We simply multiply the transpose of K matrix with Q matrix 
scores = [  
[-0.143, 0.120, 0.026,-0.300, 0.021],  
[ 0.436,-0.060,-0.035, 0.703, 0.306],  
[-0.044, 0.292, 0.045,-0.268, 0.314],  
[-0.559, 0.293, 0.076,-1.053,-0.132],  
[ 0.418,-0.007,-0.026, 0.639, 0.355]  
]

# Scale the Attention Scores 
If we didn't divide by `√dₖ`, the dot products become larger as the vector dimension grows. Large values make the softmax produce almost only `0`s and `1`s, making training unstable.
dₖ = 2  
√dₖ = √2 ≈ 1.414
scaled score =Scores / √2
so multiplying by 1/1.414 
scaled attention scores=[
[-0.101, 0.085, 0.018,-0.212, 0.015],
[ 0.308,-0.042,-0.025, 0.497, 0.216],
[-0.031, 0.206, 0.032,-0.189, 0.222],
[-0.395, 0.207, 0.054,-0.744,-0.093],
[ 0.296,-0.005,-0.018, 0.452, 0.251]
]

# Causal Mask
now we mask the upcoming inputs bcoz we pretend we dont know em
so we multiply using a mask like - [  
[0, -∞, -∞, -∞, -∞],  
[0, 0, -∞, -∞, -∞],  
[0, 0, 0, -∞, -∞],  
[0, 0, 0, 0, -∞],  
[0, 0, 0, 0, 0]  
]

Masked Scores = Scaled Scores + Mask
so masked scores = [
[-0.101,-∞,-∞,-∞,-∞],
[ 0.308,-0.042,-∞,-∞,-∞],
[-0.031,0.206,0.032,-∞,-∞],
[-0.395,0.207,0.054,-0.744,-∞],
[ 0.296,-0.005,-0.018,0.452,0.251]
]

# drop out (skipped)
skipped during text inference (generation use) and it is only used in training.

# Apply Softmax

```python
attn_weights = torch.softmax(attn_scores, dim=0)
```

so after softmax attention scores become attention weights
attention weights = [
[1.000,0,0,0,0],
[0.587,0.413,0,0,0],
[0.301,0.382,0.317,0,0],
[0.192,0.351,0.301,0.156,0],
[0.229,0.169,0.167,0.268,0.219]
]
-------------------------------------------------------------------------------------------------------------
# Multiply with V matrix
attention weights (Aw) = [
[1.000,0,0,0,0],
[0.587,0.413,0,0,0],
[0.301,0.382,0.317,0,0],
[0.192,0.351,0.301,0.156,0],
[0.229,0.169,0.167,0.268,0.219]
]

V =  
[  
[ 0.083, -0.260],  
[-0.055, 0.659],  
[ 0.517, 0.137],  
[ 0.209, -0.829],  
[-0.048, 0.562]  
]

so Aw x V= Context Vector =
[  
[ 0.083, -0.260],  
[ 0.026, 0.120],  
[ 0.170, 0.212],  
[ 0.134, 0.056],  
[ 0.127, 0.013]  
]

this is our context vector and here the story of our self attention mechanism ends