In PyTorch, an **Autograd Function** is a low-level building block that defines **how to compute forward and backward operations** for a custom computation in the automatic differentiation system.

---

### 🔧 Why it exists:

PyTorch's `autograd` system automatically tracks operations on tensors to build a computational graph, enabling backpropagation. When you need to implement **custom gradients**, you define an `autograd.Function`.

---

### 🧠 Key Concepts

#### 🔹 `torch.autograd.Function`

* You create a **subclass** of `torch.autograd.Function`.
* You implement two **static methods**:

  * `forward(ctx, ...)`: does the computation.
  * `backward(ctx, grad_output)`: computes the gradient for each input.

#### 🔹 `ctx` (Context)

* Think of it as a **"stash box"**: use it to store variables needed in `backward()` that were computed in `forward()`.

---

### ✅ Example: Custom Square Function

```python
import torch

class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # store input for backward pass
        return input * input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        return grad_input

# Use it
x = torch.tensor([2.0], requires_grad=True)
y = MySquare.apply(x)
y.backward()

print(x.grad)  # prints: tensor([4.])
```

---

### 📝 Summary Table

| Method       | Purpose                                         |
| ------------ | ----------------------------------------------- |
| `forward()`  | Computes the output from input(s)               |
| `backward()` | Computes gradients of inputs from `grad_output` |
| `ctx`        | Stores context between forward & backward       |

---

Would you like an example where a custom loss function or a non-differentiable operation is wrapped with `autograd.Function`?


