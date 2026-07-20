---
terminologies: []
prior understanding: []
---

# 🧠 LLMs From Scratch: The Master Vault

Welcome to your personal learning vault! This is the central hub where we break down the complex mathematics and architecture behind Large Language Models into simple, digestible pieces.

> [!quote] *"Your journey starts with one step"*

---

## 🏗️ Phase 1: Data Preparation
*Before an AI can read, it needs to understand how to chop up text into numbers.*

- ✂️ **[[Chapter 1 Tokenizing Text]]**
  *Breaking down sentences into smaller pieces (tokens) and mapping them to a vocabulary.*
- 🎲 **[[Chapter 2 Data Sampling]]**
  *How to efficiently feed data into our model using batches and sliding windows.*

## 🔢 Phase 2: Embeddings
*Giving meaning and structure to our raw numbers.*

- 🧠 **[[Chapter 3 Creating Token Embeddings]]**
  *Transforming basic token IDs into rich, multi-dimensional meaning vectors.*
- 📍 **[[Chapter 4 Encoding Positional Embeddings]]**
  *Giving the model a sense of order so it knows which word came first.*

## ⚙️ Phase 3: The Core Engine (Math & Attention)
*The absolute heart of modern AI: The Self-Attention Mechanism. We build this step-by-step from the ground up.*

> [!info] The Math Prerequisites
> - 📐 **[[Chapter 5.1 - Dot Products]]** — *The basic math operation for finding relationships.*
> - ✖️ **[[Chapter 5.2 - Matrix Multiplication]]** — *Doing many dot products all at once.*

> [!example] Building Self-Attention Step-by-Step
> - 🤝 **[[Chapter 5.3 - attention scores]]** — *Words asking "how much should I care about you?"*
> - ⚖️ **[[Chapter 5.4 - normalization]]** — *Using Softmax to convert raw scores into clean percentages.*
> - 🔋 **[[Chapter 5.5 - context vector]]** — *Blending everything into a super-charged vector.*
> - 🚀 **[[Chapter 5.6 - full matrix conversion]]** — *The magic trick: Doing it all instantly with Matrix Math!*

## 🧠 Phase 4: Full Architecture
*Putting all the pieces together.*

- 🧩 **[[Chapter 6 Understainding Attention Mechanism]]**
  *The grand finale. Seeing the entire Self-Attention block in action in a story format.*

## ⚙️ Phase 5: Transformer Blocks & Inference Setup
*Building the Feed-Forward Neural Network, Layer Normalization, and the final inference pipeline.*

- 🧠 **[[Chapter 7.1 FFN]]**
- ⚖️ **[[Chapter 7.2 layer normalization]]**
- 📖 **[[Chapter 7.3 transformer story]]**
- 🚀 **[[Chapter 7.4 Complete inference pipeline]]**

## 🏋️‍♂️ Phase 6: Training the Model
*Teaching our model to learn from data.*

- 🏋️ **[[Chapter 8.1 Training GPT Model]]**
- 📦 **[[Chapter 8.2 Batches and Epochs]]**
- 📊 **[[Chapter 8.3 Looking at 1 Epoch data]]**
- 📉 **[[chapter 8.4 Cross entropy loss]]**
- 🔙 **[[Chapter 8.5 Backpropogation]]**
- 🧮 **[[Chapter 8.6 Tensor]]**
- 🎯 **[[Chapter 8.7 Weight Optimization]]**

## 💾 Phase 7: Pretrained Weights & Text Generation
*Using pre-existing intelligence and generating text.*

- 💾 **[[Chapter 9.1 Saving our model weights in Pytorch]]**
- 🧠 **[[Chapter 9.2 Loading OpenAI's Weights (pretrained)]]**
- 📝 **[[Chapter 10 Text Generation]]**

## 🎓 Phase 8: Instruction Fine-Tuning
*Teaching the model to follow commands and chat.*

- 🎓 **[[Chapter 11 Instruction Fine-Tuning]]**

---

> [!tip] Vault Navigation
> **Pro Tip:** Hover over any of the chapter links above in Reading Mode to see a quick pop-over preview of its contents! You can also click the ⬅️ **Previous Step** and ➡️ **Next Step** links at the bottom of each file to read them in order like a book.
