# Quantum Neural Networks for Non-Autonomous Architectures: Research Notes

## Problem Definition
NAIS-Net is a **non-autonomous** network: each layer receives a linear transformation of the original input \( u \). This design ensures stability but creates a fundamental challenge for QNNs, which typically follow a QAOA-style alternating structure without an explicit mechanism for layer-wise external inputs.

## Proposed Circuit Solutions

### Approach A: Separate Encoding Circuits
- Allocate dedicated qubits/circuits to encode the external input \( u \) at each layer.
- **Pros**: Clean separation, minimal cross-layer interference.
- **Cons**: Qubit count scales linearly with layers → resource explosion.

### Approach B: Deep Circuit Encoding
- Encode both the layer state and the external input within the same circuit, either across time steps or across qubits.
- **Pros**: High qubit utilization.
- **Cons**: Circuit depth grows rapidly → may exceed NISQ capabilities.

## Next Steps
1. Implement a minimal 4-qubit prototype in Qiskit.
2. Compare both approaches on a simple function approximation task.
3. Measure trade-offs: circuit depth, gate count, and approximation error.

## Related Work
- [QAOA for optimization]
- [Data encoding in quantum machine learning]
