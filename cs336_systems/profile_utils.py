import torch


GIB = 1024**3


def memory_snapshot(model, optimizer=None):
    torch.cuda.synchronize()
    parameters = sum(param.numel() * param.element_size() for param in model.parameters())
    gradients = sum(param.grad.numel() * param.grad.element_size() for param in model.parameters() if param.grad is not None)
    inner_optimizer = getattr(optimizer, "optimizer", optimizer)
    optimizer_states = (
        0
        if inner_optimizer is None
        else sum(value.numel() * value.element_size() for state in inner_optimizer.state.values() for value in state.values() if torch.is_tensor(value))
    )
    allocated = torch.cuda.memory_allocated()
    result = {
        "allocated_gib": allocated / GIB,
        "phase_peak_gib": torch.cuda.max_memory_allocated() / GIB,
        "parameters_gib": parameters / GIB,
        "gradients_gib": gradients / GIB,
        "optimizer_states_gib": optimizer_states / GIB,
        "other_gib": (allocated - parameters - gradients - optimizer_states) / GIB,
    }
    torch.cuda.reset_peak_memory_stats()
    return {key: round(value, 3) for key, value in result.items()}


def summarize_memory(memory, memory_per_rank):
    max_phase_peaks = {}
    for phase in memory:
        worst = max(
            memory_per_rank,
            key=lambda item: item["phases"][phase]["phase_peak_gib"],
        )
        max_phase_peaks[phase] = {
            "rank": worst["rank"],
            "phase_peak_gib": worst["phases"][phase]["phase_peak_gib"],
        }
    run_peak_phase = max(
        max_phase_peaks,
        key=lambda phase: max_phase_peaks[phase]["phase_peak_gib"],
    )
    return {
        "per_rank": memory_per_rank,
        "max_phase_peaks": max_phase_peaks,
        "run_peak": {
            "phase": run_peak_phase,
            **max_phase_peaks[run_peak_phase],
        },
    }
