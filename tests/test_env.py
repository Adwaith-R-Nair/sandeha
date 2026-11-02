import torch


def test_cuda_runs_a_matmul():
    device = torch.device("cuda")
    a = torch.rand(1024, 1024, device=device)
    b = torch.rand(1024, 1024, device=device)

    result = a @ b

    assert result.device.type == "cuda"
    assert torch.isfinite(result).all()
