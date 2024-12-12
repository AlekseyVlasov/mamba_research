def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    
    total_size_in_bytes = total_params * 4
    total_size_in_mb = total_size_in_bytes / (1024 * 1024)
    total_size_in_gb = total_size_in_mb / 1024

    print(f"Model parameters number: {total_params}")
    print(f"Model size: {total_size_in_mb:.2f} MB")
    print(f"Model size: {total_size_in_gb:.2f} GB")
