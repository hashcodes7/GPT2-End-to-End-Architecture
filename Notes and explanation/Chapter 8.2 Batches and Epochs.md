> [!abstract] Overview
every epoch can contain many batches. every batch can contain many training sets.
simple analogy is that 
1 batch= the weights changed once
1 epoch= the entire dataset has been read through once


Suppose we have:
   - <span style="color:#FFD54F;"><b>1000 Training Sets</b></span>
   - <span style="color:#FFD54F;"><b>Batch Size = 50</b></span>

2. Number of batches:
   - **1000 ÷ 50 = 20 batches**

3. Each batch processes **50 training sets** and performs **1 weight update**.

4. Therefore:
   - **20 batches = 20 weight updates**

5. After all **20 batches** are processed, the model has seen all **1000 training sets** once. This is called **<span style="color:#4FC3F7;"><b>1 Epoch</b></span>**.

6. If we train for **10 epochs**:
   - **20 batches/epoch × 10 epochs = 200 weight updates**

``` bash
Epoch 1 

Batch 1 (Examples 1–50) 
↓ Update weights

Batch 2 (Examples 51–100)
↓ Update weights... 

Batch 20 (Examples 951–1000) ↓ 
Update weights✅ 

Entire dataset has now been seen once 
= 1 Epoch Completed (20 times weight changed)
```
