# Feedforward Neural Networks — MNIST Digit Recognition

## Project Summary

Built and compared Feedforward Neural Networks (FNN) with different activation functions and regularisation techniques on the MNIST handwritten digit dataset. This project bridges classical ML and deep learning — showing why neural networks outperform every classical algorithm and what architectural decisions drive that performance.

**Dataset:** [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)

**Business Problem:** Automatically recognise handwritten digits from images — powering cheque processing, postal code recognition, form digitization, and document processing at scale.

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| Logistic Regression (previous project) | 90.50% |
| XGBoost (previous project) | 97.05% |
| LightGBM (previous project) | 97.68% |
| FNN — ReLU (baseline) | 98.14% |
| FNN — Sigmoid | 97.15% |
| FNN — GELU | 98.26% |
| FNN — ReLU + BatchNorm | 98.18% |

**Neural networks beat every classical algorithm** — including gradient boosting which was previously the best.

---

## Network Architecture

```
Input Layer:    784 neurons  (28×28 pixels flattened)
     ↓
Hidden Layer 1: 512 neurons + ReLU + Dropout(0.2)
     ↓
Hidden Layer 2: 256 neurons + ReLU + Dropout(0.2)
     ↓
Hidden Layer 3: 128 neurons + ReLU + Dropout(0.2)
     ↓
Output Layer:   10 neurons   (digits 0-9)

Total Parameters: 567,434
```

---

## Key Learnings

### 1. Why Neural Networks Beat Classical ML

Classical algorithms make decisions directly on raw features:
```
Logistic Regression: weighted sum of pixels → prediction
XGBoost:             tree splits on pixel values → prediction
```

Neural networks learn **hierarchical representations**:
```
Layer 1 (784→512) → learns basic pixel patterns — edges, corners
Layer 2 (512→256) → learns combinations — curves, lines
Layer 3 (256→128) → learns digit parts — loops, stems, crossbars
Output  (128→10)  → combines parts → predicts digit
```

No other algorithm does this automatically. This hierarchy is why neural networks generalise better on complex data.

### 2. Why Activation Functions Exist

Without activation functions every layer is just a linear operation:
```
Layer 1: y = 2x + 1
Layer 2: y = 3x + 2
Layer 3: y = 4x + 1
Combined: y = 24x + 11 ← still one straight line
```

A 100 layer network without activations is mathematically identical to a single layer network. Activation functions add non-linearity — allowing the network to learn curves, complex boundaries, and abstract patterns.

### 3. Activation Functions Compared

**Sigmoid**
```
Range: 0 to 1
Problem: Vanishing gradients — flat curve at extremes kills backpropagation
Use: Binary classification output layer only
```

**ReLU (Rectified Linear Unit)**
```
output = max(0, x)
Range: 0 to infinity
Advantage: No vanishing gradient for positive inputs
Problem: Dying ReLU — neurons stuck at 0 for always-negative inputs
Use: Standard hidden layer activation, CNNs
```

**GELU (Gaussian Error Linear Unit)**
```
output = x × Φ(x)
Range: Similar to ReLU but smoother
Advantage: Smooth gradient flow, small negative signals pass through
Use: Transformers, LLMs, modern architectures
```

**Results on MNIST:**
```
GELU:    98.26% ← best
ReLU:    98.14%
Sigmoid: 97.15% ← worst — vanishing gradients
```

At scale:
```
Processing 1,000,000 digits daily:
GELU    → 17,400 errors
Sigmoid → 28,500 errors
11,100 extra errors daily from wrong activation choice
```

### 4. Backpropagation — How The Network Learns

Training has two phases every batch:

**Forward pass** — data flows forward, prediction made, error calculated
**Backward pass** — error flows backward, gradients calculated, weights updated

```
Weight update rule:
new_weight = old_weight - learning_rate × gradient

If gradient positive → weight too high → decrease it
If gradient negative → weight too low  → increase it
```

PyTorch handles all of this automatically:
```python
loss.backward()      # calculate all 567,434 gradients
optimizer.step()     # update all 567,434 weights
optimizer.zero_grad() # reset gradients for next batch
```

Training progress — loss dropping every epoch confirms backpropagation working:
```
Epoch 1  → Loss: 0.4376 | Accuracy: 87.21%
Epoch 2  → Loss: 0.1502 | Accuracy: 95.56%
Epoch 5  → Loss: 0.0627 | Accuracy: 98.09%
Epoch 10 → Loss: 0.0309 | Accuracy: 99.01%
```

### 5. Dropout — Preventing Overfitting

Dropout randomly disables a percentage of neurons during each training batch:

```python
nn.Dropout(0.2)  # disable 20% of neurons randomly each batch
```

```
Without Dropout:
Training accuracy: 99.5%
Test accuracy:     97.0%
Gap:               2.5% ← overfitting

With Dropout(0.2):
Training accuracy: 99.01%
Test accuracy:     98.27%
Gap:               0.74% ← much better generalisation
```

Dropout forces the network to learn redundant representations — no single neuron can be relied upon. This makes the network robust to new data.

### 6. Batch Normalisation

Batch Normalisation normalises the output of each layer back to mean=0, std=1 before passing to the next layer — fixing Internal Covariate Shift:

```
Without BatchNorm:
Input layer:   values 0-1
After Layer 1: values -5 to 8
After Layer 2: values -50 to 200  ← distribution shifting
After Layer 3: values -500 to 2000 ← exploding

With BatchNorm:
Every layer output → normalised back to mean=0, std=1
Each layer always receives stable, consistent input
```

**Results on MNIST:**
```
ReLU (baseline):  98.14%
ReLU + BatchNorm: 98.18%
Difference:       +0.04% — minimal on shallow network
```

BatchNorm's real power shows on very deep networks (20+ layers) and complex datasets — enabling 5-10x faster training and allowing higher learning rates. On 3-layer MNIST the distribution shift problem barely exists.

### 7. DataLoader — Why Batching Matters

Training on all 60,000 images at once:
- Memory: Would require gigabytes of GPU memory
- Speed: One weight update per epoch — extremely slow
- Quality: Too smooth gradients — slow learning

DataLoader splits data into batches:
```python
DataLoader(dataset, batch_size=256, shuffle=True)

60,000 images / 256 batch size = 235 weight updates per epoch
10 epochs × 235 updates = 2,350 total weight updates
```

**shuffle=True** randomises order every epoch — prevents model memorising data order and overfitting to sequence patterns.

### 8. Adam Optimizer

Adam (Adaptive Moment Estimation) automatically adjusts the learning rate for each weight:

```
Standard gradient descent: same learning rate for all weights
Adam: different learning rate per weight based on history

Weights updated rarely → Adam gives them larger steps
Weights updated often  → Adam gives them smaller steps
```

This is why Adam converges faster than plain gradient descent — it adapts to the landscape of each individual weight.

### 9. The 567,434 Parameters

```
Layer 1: 784 × 512 + 512 bias = 401,920 parameters
Layer 2: 512 × 256 + 256 bias = 131,328 parameters
Layer 3: 256 × 128 + 128 bias =  32,896 parameters
Output:  128 × 10  + 10  bias =   1,290 parameters
Total:                           567,434 parameters
```

Each parameter adjusted 2,350 times through backpropagation to achieve 98.27% accuracy.

---

## Tools & Libraries

```python
datasets          # Hugging Face dataset loading
numpy             # Numerical operations
torch             # PyTorch deep learning framework
torch.nn          # Neural network layers
torch.optim       # Optimizers (Adam)
torch.utils.data  # DataLoader, TensorDataset
scikit-learn      # Evaluation metrics
matplotlib        # Visualisation
Pillow            # Image processing
python-dotenv     # Environment variable management
huggingface_hub   # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/neural-networks.git
cd neural-networks

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run
python neural_network.py
```

---

## PM Perspective

### Why Neural Networks Change The Product Equation

Every classical ML model you build has a ceiling. Decision trees plateau. Gradient boosting plateaus. The patterns they can learn are fundamentally limited by their architecture.

Neural networks have no ceiling — add more layers, more neurons, more data and they keep improving. This is why every major AI product — GPT, Gemini, Stable Diffusion, AlphaFold — is built on neural networks.

As a PM understanding where that ceiling is for each algorithm helps you make the right build vs buy, build vs fine-tune decisions.

### Architectural Decisions Are Product Decisions

Every choice in the network architecture has a business consequence:

**Activation function:**
```
Wrong choice (Sigmoid in hidden layers) → 11,100 extra errors per day
Right choice (GELU) → state of the art performance
Cost of getting this wrong: real, measurable, preventable
```

**Dropout rate:**
```
Too low  → model overfits → great in testing, poor in production
Too high → model underfits → poor everywhere
Right rate → consistent performance across training and production
```

**Number of layers and neurons:**
```
Too few → model can't learn complex patterns → low accuracy
Too many → model overfits, slow inference, high serving cost
Right size → best accuracy at acceptable inference cost
```

These are not purely engineering decisions. They require a PM to define:
- What is the minimum acceptable accuracy?
- What is the maximum acceptable inference latency?
- What is the infrastructure cost budget?

### The 567,434 Parameters In Production

```
Model size:      ~2MB — deployable on mobile devices
Inference time:  milliseconds per prediction
Training time:   ~5 minutes on CPU
Monthly retrain: low cost — can retrain on new data regularly
```

This model could power:
- Mobile cheque scanning — bank processes millions daily
- Postal code recognition — logistics companies scan millions of parcels
- Form digitization — government offices processing handwritten forms
- Handwritten order processing — retail and restaurant chains

### What To Monitor After Launch

**Accuracy drift** — handwriting styles vary by region, age, culture. A model trained on US handwriting may underperform on Indian handwriting. Monitor accuracy by user segment.

**Confidence distribution** — track the softmax probability of the predicted class. If average confidence drops over time the model is becoming less certain — signal to retrain.

**Error patterns** — which digits are being misclassified? If digit 5 error rate spikes after a product update something changed in the input pipeline.

**Latency** — 567,434 parameters is lightweight now. As you add features and users scale, inference latency must be monitored against SLA.

### Fine-tuning vs Training From Scratch

This network was trained from scratch on MNIST. In production you would almost never do this for images — you would fine-tune a pre-trained model:

```
Train from scratch: weeks of compute, millions of images needed
Fine-tune ResNet:   hours of compute, thousands of images sufficient
Fine-tune ViT:      state of the art with minimal data
```

Understanding when to train from scratch vs fine-tune is one of the highest leverage decisions an AI PM makes — it determines project timeline, compute budget, and team size.

### Key Insight

> A neural network with the wrong activation function makes 11,100 more errors per day than one with the right activation function. The data scientist chooses the function. The PM defines what error rate is acceptable. Both decisions are required.

---


---

*Built as part of a learning journey from Senior PM to AI PM — April 2026*
