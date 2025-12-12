import torch
import torch.multiprocessing as mp
import time
import numpy as np
from typing import Tuple
import pickle
import os


def create_ipc_handle(tensor: torch.Tensor) -> Tuple:
    """
    创建CUDA IPC句柄
    返回: (device_index, handle_bytes, size_bytes)
    """
    storage = tensor.untyped_storage()
    return storage._share_cuda_()


def inspect_ipc_handle_size(tensor: torch.Tensor):
    """
    检查 IPC handle 的大小
    """
    handle = create_ipc_handle(tensor)
    
    print(f"\n=== IPC Handle 大小分析 ===")
    print(f"Handle 类型: {type(handle)}")
    print(f"Handle 长度: {len(handle) if hasattr(handle, '__len__') else 'N/A'}")
    
    if isinstance(handle, tuple):
        total_size = 0
        for i, item in enumerate(handle):
            if isinstance(item, bytes):
                size = len(item)
                print(f"  Element {i} (bytes): {size} bytes")
                total_size += size
            elif isinstance(item, int):
                size = 8  # Python int 通常是 8 bytes (64-bit)
                print(f"  Element {i} (int): {item}, 大小: {size} bytes")
                total_size += size
            else:
                size = len(pickle.dumps(item)) if hasattr(pickle, 'dumps') else 'unknown'
                print(f"  Element {i} ({type(item).__name__}): 大小: {size}")
                if isinstance(size, int):
                    total_size += size
        
        print(f"\n总大小: {total_size} bytes ({total_size/1024:.2f} KB)")
        
        # 尝试序列化整个 handle 看看大小
        try:
            serialized = pickle.dumps(handle)
            print(f"序列化后大小: {len(serialized)} bytes ({len(serialized)/1024:.2f} KB)")
        except Exception as e:
            print(f"序列化失败: {e}")
    
    return handle


def reconstruct_from_handle(handle_tuple: Tuple) -> torch.Tensor:
    """
    从IPC句柄重建Tensor
    """
    # 使用 * 解包，因为 _share_cuda_() 返回的元组长度可能因PyTorch版本而异
    # _new_shared_cuda 可以接受可变参数
    new_storage = torch.UntypedStorage._new_shared_cuda(*handle_tuple)
    
    # 重建Tensor（需要知道原始shape和dtype）
    # 注意：这里只重建存储，shape和dtype需要另外传递
    return new_storage


def producer(queue: mp.Queue, event: mp.Event, tensor_size: int = 1000, device_id: int = 0):
    """
    生产者进程：创建CUDA Tensor并共享
    
    参数:
        device_id: 使用的GPU设备ID（支持跨卡传输）
    """
    print(f"[Producer] PID: {os.getpid()}, CUDA available: {torch.cuda.is_available()}")
    
    # 创建CUDA Tensor（可以指定不同的GPU）
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    print(f"[Producer] Using device: {device}")
    
    # 创建要共享的数据
    original_tensor = torch.randn(tensor_size, tensor_size, device=device)
    original_tensor = original_tensor * 2 + 1  # 做一些变换以便验证
    
    print(f"[Producer] Original tensor created, shape: {original_tensor.shape}")
    print(f"[Producer] Tensor mean: {original_tensor.mean().item():.4f}, std: {original_tensor.std().item():.4f}")
    
    # 创建IPC句柄
    ipc_handle = create_ipc_handle(original_tensor)
    
    # 准备传递给消费者的数据
    metadata = {
        'shape': original_tensor.shape,
        'dtype': original_tensor.dtype,
        'device': original_tensor.device,
        'mean': original_tensor.mean().item(),
        'std': original_tensor.std().item()
    }
    
    # 将句柄和元数据放入队列
    queue.put((ipc_handle, metadata))
    
    # 等待消费者确认接收
    print("[Producer] Waiting for consumer to receive...")
    event.wait(timeout=10)
    
    # 验证消费者是否修改了数据
    time.sleep(1)  # 给消费者时间修改
    
    # 检查原始tensor是否被修改
    new_mean = original_tensor.mean().item()
    print(f"[Producer] After consumer modification, mean: {new_mean:.4f}")
    
    # 做一些额外的修改，看消费者是否能看到
    original_tensor.add_(5.0)
    print(f"[Producer] Added 5.0 to tensor, new mean: {original_tensor.mean().item():.4f}")
    
    time.sleep(2)  # 让消费者读取修改后的值
    print("[Producer] Done")


def consumer(queue: mp.Queue, event: mp.Event, target_device_id: int = None):
    """
    消费者进程：从IPC句柄重建Tensor
    
    参数:
        target_device_id: 目标GPU设备ID（None表示使用metadata中的设备，支持跨卡传输）
    """
    print(f"[Consumer] PID: {os.getpid()}, CUDA available: {torch.cuda.is_available()}")
    
    # 从队列获取IPC句柄和元数据
    print("[Consumer] Waiting for IPC handle...")
    ipc_handle, metadata = queue.get(timeout=10)
    
    print(f"[Consumer] Received metadata: shape={metadata['shape']}, dtype={metadata['dtype']}")
    print(f"[Consumer] Source device: {metadata['device']}")
    
    # 获取源设备ID
    source_device_idx = metadata['device'].index if hasattr(metadata['device'], 'index') else 0
    source_device = torch.device(f"cuda:{source_device_idx}")
    
    # 确定目标设备
    if target_device_id is not None:
        target_device = torch.device(f"cuda:{target_device_id}")
        is_cross_gpu = (target_device_id != source_device_idx)
        print(f"[Consumer] Target device: {target_device} ({'跨卡传输' if is_cross_gpu else '同卡传输'})")
    else:
        target_device = source_device
        is_cross_gpu = False
        print(f"[Consumer] Using source device: {target_device} (同卡传输)")
    
    # 关键：IPC handle 创建的存储必须在源设备上打开
    # 使用 torch.cuda.device 上下文管理器确保在正确的设备上操作
    with torch.cuda.device(source_device):
        # 从句柄重建存储（在源设备上）
        new_storage = reconstruct_from_handle(ipc_handle)
        
        # 在源设备上创建tensor
        source_tensor = torch.tensor(
            [], 
            dtype=metadata['dtype'],
            device=source_device
        ).set_(new_storage, 0, metadata['shape'])
    
    # 如果跨卡，需要复制到目标设备
    if is_cross_gpu:
        print(f"[Consumer] Copying tensor from {source_device} to {target_device}")
        shared_tensor = source_tensor.to(target_device, non_blocking=True)
        # 清理源设备上的tensor
        del source_tensor
        torch.cuda.empty_cache()
    else:
        # 同卡传输，直接使用
        shared_tensor = source_tensor
    
    print(f"[Consumer] Reconstructed tensor, shape: {shared_tensor.shape}")
    print(f"[Consumer] Initial mean: {shared_tensor.mean().item():.4f}, "
          f"expected: {metadata['mean']:.4f}")
    
    # 验证数据是否正确
    assert torch.allclose(
        shared_tensor.mean(), 
        torch.tensor(metadata['mean'], device=shared_tensor.device),
        rtol=1e-4
    ), "Data mismatch after IPC transfer!"
    
    print("[Consumer] Data verification passed!")
    
    # 修改共享数据
    shared_tensor.mul_(2.0)
    print(f"[Consumer] Multiplied tensor by 2.0, new mean: {shared_tensor.mean().item():.4f}")
    
    # 通知生产者已接收并修改
    event.set()
    
    # 等待并检查生产者的修改
    time.sleep(2)
    print(f"[Consumer] After producer's addition, mean: {shared_tensor.mean().item():.4f}")
    
    # 验证双向修改都生效
    expected_mean = (metadata['mean'] * 2) + 5.0  # 消费者*2，生产者+5
    actual_mean = shared_tensor.mean().item()
    print(f"[Consumer] Expected mean after all modifications: {expected_mean:.4f}")
    print(f"[Consumer] Actual mean: {actual_mean:.4f}")
    
    # 清理
    del shared_tensor
    torch.cuda.empty_cache()
    
    print("[Consumer] Done")


def test_cuda_ipc_basic():
    """基础CUDA IPC测试（同卡传输）"""
    print("=" * 60)
    print("Testing basic CUDA IPC between processes (same GPU)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # 创建进程间通信对象
    queue = mp.Queue()
    event = mp.Event()
    
    # 启动进程（都使用 GPU 0）
    producer_process = mp.Process(
        target=producer,
        args=(queue, event, 500, 0)  # 500x500 tensor, GPU 0
    )
    
    consumer_process = mp.Process(
        target=consumer,
        args=(queue, event, None)  # None 表示使用源设备（同卡）
    )
    
    try:
        producer_process.start()
        consumer_process.start()
        
        producer_process.join(timeout=30)
        consumer_process.join(timeout=30)
        
        assert not producer_process.is_alive(), "Producer process timed out!"
        assert not consumer_process.is_alive(), "Consumer process timed out!"
        
        print("\n✅ Basic CUDA IPC test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
    finally:
        # 确保清理
        if producer_process.is_alive():
            producer_process.terminate()
        if consumer_process.is_alive():
            consumer_process.terminate()


def test_cuda_ipc_cross_gpu():
    """测试CUDA IPC跨卡传输"""
    print("\n" + "=" * 60)
    print("Testing CUDA IPC cross-GPU transfer")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"⚠️  只有 {num_gpus} 个GPU，需要至少2个GPU才能测试跨卡传输")
        print("跳过跨卡传输测试")
        return
    
    print(f"检测到 {num_gpus} 个GPU")
    
    # 检查 GPU 0 和 GPU 1 之间的 P2P 访问
    can_p2p_0_to_1 = torch.cuda.can_device_access_peer(0, 1)
    can_p2p_1_to_0 = torch.cuda.can_device_access_peer(1, 0)
    
    print(f"GPU 0 -> GPU 1 P2P访问: {can_p2p_0_to_1}")
    print(f"GPU 1 -> GPU 0 P2P访问: {can_p2p_1_to_0}")
    
    if not (can_p2p_0_to_1 or can_p2p_1_to_0):
        print("⚠️  GPU之间不支持P2P访问，跨卡IPC可能失败")
        print("注意：即使不支持P2P，CUDA IPC也可能工作（通过PCIe）")
    
    # 创建进程间通信对象
    queue = mp.Queue()
    event = mp.Event()
    
    # 测试：GPU 0 -> GPU 1
    print(f"\n测试：GPU 0 (Producer) -> GPU 1 (Consumer)")
    
    producer_process = mp.Process(
        target=producer,
        args=(queue, event, 500, 0)  # GPU 0 创建tensor
    )
    
    consumer_process = mp.Process(
        target=consumer,
        args=(queue, event, 1)  # GPU 1 接收（跨卡）
    )
    
    try:
        producer_process.start()
        consumer_process.start()
        
        producer_process.join(timeout=30)
        consumer_process.join(timeout=30)
        
        assert not producer_process.is_alive(), "Producer process timed out!"
        assert not consumer_process.is_alive(), "Consumer process timed out!"
        
        print("\n✅ Cross-GPU CUDA IPC test passed!")
        
    except Exception as e:
        print(f"\n❌ Cross-GPU test failed with error: {e}")
        print("这可能是因为GPU之间不支持P2P访问，或者CUDA IPC限制")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理
        if producer_process.is_alive():
            producer_process.terminate()
        if consumer_process.is_alive():
            consumer_process.terminate()


def multi_tensor_producer(queue: mp.Queue):
    """多tensor生产者函数（模块级别，可被pickle）"""
    torch.cuda.set_device(0)
    
    # 创建多个不同形状和类型的tensor
    # 使用 randn + 偏移量来生成非0均值的tensor
    tensors = [
        torch.randn(48000, 1536, device='cuda') * 2.0 + 5.0,  # float32, mean约等于5.0
    ]
    
    # 为每个tensor创建IPC句柄
    ipc_data = []
    for i, tensor in enumerate(tensors):
        handle = create_ipc_handle(tensor)
        mean_value = tensor.mean().item() if tensor.is_floating_point() else 0
        metadata = {
            'index': i,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device,
            'mean': mean_value
        }
        ipc_data.append((handle, metadata))
        
        # 打印 producer 的 tensor 信息
        if tensor.is_floating_point():
            print(f"[Producer] Tensor {i}: shape={tensor.shape}, mean={mean_value:.4f}")
        else:
            print(f"[Producer] Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    queue.put(ipc_data)
    print(f"[Producer] Sent {len(tensors)} tensors via IPC")


def multi_tensor_consumer(queue: mp.Queue):
    """多tensor消费者函数（模块级别，可被pickle）"""
    torch.cuda.set_device(0)
    
    ipc_data = queue.get(timeout=10)
    reconstructed_tensors = []
    
    for handle, metadata in ipc_data:
        # 重建存储
        storage = reconstruct_from_handle(handle)
        
        # 重建tensor
        tensor = torch.tensor(
            [], 
            dtype=metadata['dtype'],
            device=metadata['device']
        ).set_(storage, 0, metadata['shape'])
        
        reconstructed_tensors.append(tensor)
        
        # 验证
        if tensor.is_floating_point():
            actual_mean = tensor.mean().item()
            expected_mean = metadata['mean']
            print(f"[Consumer] Tensor {metadata['index']}: shape={tensor.shape}, "
                  f"expected_mean={expected_mean:.4f}, actual_mean={actual_mean:.4f}")
        else:
            print(f"[Consumer] Tensor {metadata['index']}: shape={tensor.shape}, "
                  f"dtype={tensor.dtype}")

    
    print(f"[Consumer] Successfully reconstructed {len(reconstructed_tensors)} tensors")


def test_cuda_ipc_multiple_tensors():
    """测试多个Tensor的IPC共享"""
    print("\n" + "=" * 60)
    print("Testing CUDA IPC with multiple tensors")
    print("=" * 60)
    
    # 运行测试
    queue = mp.Queue()
    
    p1 = mp.Process(target=multi_tensor_producer, args=(queue,))
    p2 = mp.Process(target=multi_tensor_consumer, args=(queue,))
    
    p1.start()
    p2.start()
    
    p1.join(timeout=20)
    p2.join(timeout=20)
    
    print("\n✅ Multiple tensor IPC test passed!")


def performance_producer(queue: mp.Queue, size_mb: int = 100):
    """性能测试生产者函数（模块级别，可被pickle）"""
    torch.cuda.set_device(0)
    
    # 创建指定大小的tensor
    elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    tensor = torch.randn(elements, device='cuda')
    
    print(f"[Producer] Creating {size_mb}MB tensor ({elements} elements)")
    
    # 预热
    for _ in range(3):
        _ = create_ipc_handle(tensor)
    
    # 计时
    start_time = time.time()
    handle = create_ipc_handle(tensor)
    
    ipc_time = time.time() - start_time
    
    metadata = {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'size_mb': size_mb,
        'ipc_time': ipc_time
    }
    
    queue.put((handle, metadata))
    print(f"[Producer] IPC handle creation time: {ipc_time*1000:.2f}ms")


def performance_consumer(queue: mp.Queue):
    """性能测试消费者函数（模块级别，可被pickle）"""
    torch.cuda.set_device(0)
    
    handle, metadata = queue.get(timeout=10)
    
    # 预热
    for _ in range(3):
        _ = reconstruct_from_handle(handle)
    
    # 计时重建
    start_time = time.time()
    storage = reconstruct_from_handle(handle)
    tensor = torch.tensor([], dtype=metadata['dtype'], device='cuda'
                        ).set_(storage, 0, metadata['shape'])
    recon_time = time.time() - start_time
    
    bandwidth = metadata['size_mb'] / recon_time  # MB/s
    
    print(f"[Consumer] Tensor reconstruction time: {recon_time*1000:.2f}ms")
    print(f"[Consumer] Effective bandwidth: {bandwidth:.2f} MB/s")
    print(f"[Consumer] Total IPC latency: {(metadata['ipc_time'] + recon_time)*1000:.2f}ms")


def test_cuda_ipc_performance():
    """CUDA IPC性能测试"""
    print("\n" + "=" * 60)
    print("Testing CUDA IPC performance")
    print("=" * 60)
    
    # 测试不同大小的数据传输
    for size_mb in [10, 100, 500]:
        print(f"\nTesting {size_mb}MB tensor...")
        
        queue = mp.Queue()
        
        p1 = mp.Process(target=performance_producer, args=(queue, size_mb))
        p2 = mp.Process(target=performance_consumer, args=(queue,))
        
        p1.start()
        p2.start()
        
        p1.join(timeout=30)
        p2.join(timeout=30)


def test_ipc_handle_size():
    """测试 IPC handle 的大小"""
    print("\n" + "=" * 60)
    print("Testing IPC Handle Size")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    torch.cuda.set_device(0)
    
    # 测试不同大小的 tensor
    test_tensors = [
        ("Small", torch.randn(100, 100, device='cuda')),
        ("Medium", torch.randn(48000, 1536, device='cuda')),
        ("Large", torch.randn(1000, 1000, device='cuda')),
    ]
    
    for name, tensor in test_tensors:
        print(f"\n--- {name} Tensor ({tensor.shape}, {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB) ---")
        handle = inspect_ipc_handle_size(tensor)
        print()


def main():
    """运行所有CUDA IPC测试"""
    print("Starting CUDA IPC tests...")
    
    # 设置多进程启动方法（对于CUDA很重要）
    mp.set_start_method('spawn', force=True)
    
    # 运行测试
    try:
        test_ipc_handle_size()  # 先测试 handle 大小
        test_cuda_ipc_basic()  # 同卡传输测试
        test_cuda_ipc_cross_gpu()  # 跨卡传输测试
        test_cuda_ipc_multiple_tensors()
        test_cuda_ipc_performance()
        
        print("\n" + "=" * 60)
        print("All CUDA IPC tests completed successfully! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # 设置环境变量（可选）
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    exit_code = main()
    exit(exit_code)
