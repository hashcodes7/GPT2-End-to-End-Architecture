
Incoming - layernorm Vector=
[  
[ 1.40,-1.52, 0.12],  
[-1.15, 1.29,-0.14],  
[ 0.71, 0.70,-1.41],  
[ 1.31,-1.14,-0.17],  
[-0.16, 0.91,-0.75]  
]

we will do 3 things -> Linear Layer (W₁) -> GELU (Activation Function) -> Linear Layer 

# 1 Top Linear Layer W₁
Here we expand features of our Layer normed vector because it only has 3 features.
W₁ (3×6) =  (ingredient)
[  
[ 0.4, -0.2, 0.1, 0.6, -0.5, 0.3],  
[-0.1, 0.8, -0.4, 0.2, 0.7, -0.6],  
[ 0.5, 0.3, 0.9, -0.2, 0.1, 0.4]  
]
x
layernorm Vector=
[  
[ 1.40,-1.52, 0.12],  
[-1.15, 1.29,-0.14],  
[ 0.71, 0.70,-1.41],  
[ 1.31,-1.14,-0.17],  
[-0.16, 0.91,-0.75]  
]

FFN Hidden Layer (Before GELU)  
[  
[ 0.772, -1.460, 0.856, 0.512, -1.752, 1.428],  
[-0.644, 1.219, -0.757, -0.404, 1.463, -1.180],  
[-0.492, 0.118, -1.478, -0.566, -0.006, -0.891],  
[ 0.551, -1.228, 0.438, 0.575, -1.471, 1.012],  
[-0.529, 0.536, -1.055, 0.236, 0.640, -0.903]  
]

# FFN -2 GELU activation
![[Pasted image 20260627200626.png|423]]
for each value in matrix we are not letting it get too negative and normalizing it of sorts.
after applying to gelu to each value, some values eg :
GELU(0.772) ≈ 0.603  
GELU(-1.460) ≈ -0.106  
GELU(0.856) ≈ 0.689  
GELU(-1.752) ≈ -0.070
The result becomes :
FFN Hidden Layer (After GELU)=
[
 [ 0.603, -0.106,  0.689,  0.357, -0.070,  1.319],
 [-0.167,  1.082, -0.170, -0.139,  1.354, -0.141],
 [-0.153,  0.064, -0.103, -0.161, -0.003, -0.167],
 [ 0.391, -0.135,  0.293,  0.414, -0.105,  0.854],
 [-0.158,  0.378, -0.154,  0.140,  0.473, -0.165]
]

# 3- Second FFN Linear Layer (`W₂`)
FFN Output = GELU Output (above) × W₂
W₂ (6×3) =   (ingredient)
[  
[ 0.2, -0.4, 0.5],  
[-0.3, 0.6, 0.1],  
[ 0.7, -0.2, -0.5],  
[ 0.1, 0.3, 0.8],  
[-0.6, 0.5, 0.2],  
[ 0.4, -0.1, 0.3]  
]


FFN Output = 
[
 [ 1.228, -0.564,  0.583],
 [-0.810,  1.420, -0.561],
 [-0.210,  0.117, -0.015],
 [ 0.836, -0.351,  0.390],
 [-0.339,  0.563, -0.140]
]
This is the final output that comes out of FFN
