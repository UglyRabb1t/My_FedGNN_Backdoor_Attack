"""
Hessian Analysis Module for GNN Models
=======================================
Provides Hessian trace and eigenvalue estimation for graph neural networks
using Hutchinson's estimator and power iteration method.
"""
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time


def get_params_with_grad(model):
    """
    Get model parameters that have gradients (excluding None gradients)
    """
    params = []
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            params.append(p)
            grads.append(p.grad)
    return params, grads


def compute_hessian_trace_hutchinson(model, data_loader, loss_func, device,
                                   num_samples=50, max_graphs_per_batch=10):
    """
    Compute Hessian trace using Hutchinson's estimator.

    Args:
        model: GNN model
        data_loader: DataLoader providing (batch_graphs, batch_labels)
        loss_func: Loss function
        device: torch device
        num_samples: Number of random projections for trace estimation
        max_graphs_per_batch: Maximum number of graphs to use for trace computation

    Returns:
        Estimated Hessian trace (float), or None if computation fails
    """
    model.eval()
    trace_estimate = 0.0
    num_graphs_used = 0

    # Store original precision and move to full precision for Hessian computation
    original_dtype = next(model.parameters()).dtype
    model = model.to(torch.float32)

    # Collect a subset of graphs for computation
    batch_graphs_list = []
    batch_labels_list = []

    for batch_graphs, batch_labels in data_loader:
        batch_graphs_list.append(batch_graphs)
        batch_labels_list.append(batch_labels)
        num_graphs_used += len(batch_labels)
        if num_graphs_used >= max_graphs_per_batch:
            break

    if num_graphs_used == 0:
        print("[Hessian] Warning: No data available for trace computation")
        return None

    # Combine all batches - use dgl.batch for DGLGraph objects
    batch_graphs = dgl.batch(batch_graphs_list)
    batch_labels = torch.cat(batch_labels_list)

    # Ensure all data is in full precision
    batch_graphs = batch_graphs.to(device)
    batch_labels = batch_labels.to(torch.long).to(device)  # Ensure labels are long type
    batch_x = batch_graphs.ndata['feat'].to(torch.float32).to(device)  # Convert features to float32
    batch_e = batch_graphs.edata['feat'].to(torch.float32).to(device)  # Convert edge features to float32

    # Forward pass
    output = model(batch_graphs, batch_x, batch_e)
    loss = loss_func(output, batch_labels)

    # First backward pass to get gradients using autograd.grad to avoid memory leak
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    # Update model gradients manually
    for p, g in zip(model.parameters(), grads):
        if p.grad is not None:
            p.grad = g

    # Get parameters with valid gradients
    params, grads = get_params_with_grad(model)

    if len(params) == 0:
        print("[Hessian] Warning: No parameters with gradients found")
        return None

    # Hutchinson's estimator: trace(H) ≈ E[v^T H v]
    for i in range(num_samples):
        # Generate random vector v (Rademacher distribution: ±1)
        v = [torch.randint(0, 2, g.shape).float().to(device) * 2 - 1 for g in grads]

        try:
            # Compute Hv = Hessian * v
            hv = torch.autograd.grad(
                grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
            )
            # Compute v^T * Hv
            product = sum([torch.sum(v_i * h_i) for v_i, h_i in zip(v, hv) if h_i is not None])
            trace_estimate += product.item()
        except Exception as e:
            print(f"[Hessian] Sampling {i+1}/{num_samples} failed: {e}")
            return None

    trace_estimate = trace_estimate / num_samples

    # Restore original precision
    model = model.to(original_dtype)

    return trace_estimate


def compute_hessian_eigenvalue_power(model, data_loader, loss_func, device,
                                     max_iter=100, max_graphs_per_batch=10,
                                     verbose=True):
    """
    Compute the largest Hessian eigenvalue using power iteration method.

    Args:
        model: GNN model
        data_loader: DataLoader providing (batch_graphs, batch_labels)
        loss_func: Loss function
        device: torch device
        max_iter: Maximum number of power iterations
        max_graphs_per_batch: Maximum number of graphs to use for computation
        verbose: Whether to print iteration progress

    Returns:
        Estimated largest eigenvalue (float), or None if computation fails
    """
    model.eval()

    # Store original precision and move to full precision for Hessian computation
    original_dtype = next(model.parameters()).dtype
    model = model.to(torch.float32)

    # Collect a subset of graphs for computation
    batch_graphs_list = []
    batch_labels_list = []
    num_graphs_used = 0

    for batch_graphs, batch_labels in data_loader:
        batch_graphs_list.append(batch_graphs)
        batch_labels_list.append(batch_labels)
        num_graphs_used += len(batch_labels)
        if num_graphs_used >= max_graphs_per_batch:
            break

    if num_graphs_used == 0:
        print("[Hessian] Warning: No data available for eigenvalue computation")
        return None

    # Combine all batches - use dgl.batch for DGLGraph objects
    batch_graphs = dgl.batch(batch_graphs_list)
    batch_labels = torch.cat(batch_labels_list)

    # Ensure all data is in full precision
    batch_graphs = batch_graphs.to(device)
    batch_labels = batch_labels.to(torch.long).to(device)  # Ensure labels are long type
    batch_x = batch_graphs.ndata['feat'].to(torch.float32).to(device)  # Convert features to float32
    batch_e = batch_graphs.edata['feat'].to(torch.float32).to(device)  # Convert edge features to float32

    # Get parameters with valid gradients
    params = [p for p in model.parameters() if p.requires_grad]
    grads_list = []

    # Initialize random vector v
    v = [torch.randn_like(p).to(device) for p in params]
    v = [vi / (torch.norm(vi) + 1e-10) for vi in v]

    eigenvalue = 0.0

    for i in range(max_iter):
        # Forward pass
        output = model(batch_graphs, batch_x, batch_e)
        loss = loss_func(output, batch_labels)

        # Backward pass
        loss.backward(create_graph=True, retain_graph=True)

        # Get current gradients
        grads = [p.grad for p in params]

        try:
            # Compute Hv = Hessian * v
            hv = torch.autograd.grad(
                grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
            )
        except Exception as e:
            print(f"[Hessian] Eigenvalue iteration {i+1} failed: {e}")
            return None

        # Rayleigh quotient: λ = (v^T Hv) / (v^T v)
        numerator = sum([torch.sum(v_i * h_i) for v_i, h_i in zip(v, hv) if h_i is not None])
        denominator = sum([torch.sum(v_i * v_i) for v_i in v])
        eigenvalue = (numerator / (denominator + 1e-10)).item()

        # Update v
        v = [h_i / (torch.norm(h_i) + 1e-10) if h_i is not None else torch.randn_like(v_i).to(device)
             for h_i, v_i in zip(hv, v)]

        if verbose and (i + 1) % 20 == 0:
            print(f"[Hessian] Eigenvalue iteration {i+1}/{max_iter}: λ ≈ {eigenvalue:.6f}")

    # Restore original precision
    model = model.to(original_dtype)

    return eigenvalue


def compute_hessian_metrics(model, data_loader, loss_func, device, args):
    """
    Compute both Hessian trace and eigenvalue based on command line arguments.

    Args:
        model: GNN model
        data_loader: DataLoader
        loss_func: Loss function
        device: torch device
        args: Arguments containing hessian computation settings

    Returns:
        Dictionary with computed metrics (keys: 'trace', 'eigenvalue')
    """
    metrics = {}

    if not hasattr(args, 'compute_hessian') or not args.compute_hessian:
        return metrics

    print("\n" + "=" * 60)
    print("HESSIAN ANALYSIS")
    print("=" * 60)

    start_time = time.time()

    # Compute Hessian trace
    if args.compute_hessian_trace:
        print("[Hessian] Computing trace using Hutchinson's estimator...")
        trace = compute_hessian_trace_hutchinson(
            model=model,
            data_loader=data_loader,
            loss_func=loss_func,
            device=device,
            num_samples=args.hessian_trace_samples,
            max_graphs_per_batch=args.hessian_batch_size
        )
        if trace is not None:
            metrics['trace'] = trace
            print(f"[Hessian] Hessian trace: {trace:.4f}")
        else:
            print("[Hessian] Trace computation failed")

    # Compute largest eigenvalue (optional, can be slow)
    if args.compute_hessian_eigenvalue:
        print("\n[Hessian] Computing largest eigenvalue using power iteration...")
        eigenvalue = compute_hessian_eigenvalue_power(
            model=model,
            data_loader=data_loader,
            loss_func=loss_func,
            device=device,
            max_iter=args.hessian_eigenvalue_iter,
            max_graphs_per_batch=args.hessian_batch_size,
            verbose=False
        )
        if eigenvalue is not None:
            metrics['eigenvalue'] = eigenvalue
            print(f"[Hessian] Largest eigenvalue: {eigenvalue:.4f}")
        else:
            print("[Hessian] Eigenvalue computation failed")

    elapsed_time = time.time() - start_time
    print(f"[Hessian] Computation time: {elapsed_time:.2f}s")
    print("=" * 60)

    return metrics
