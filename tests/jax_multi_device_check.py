# jax_multi_device_check.py
import jax
import jax.numpy as jnp
import os
import jax.debug 

def _parallel_computation_implementation(local_data_slice_per_device):
    device_idx_in_pmap = jax.lax.axis_index('data_parallel_axis')
    
    # Sử dụng jax.debug.print để in shape trong quá trình JAX tracing
    jax.debug.print("Shape of local_data_slice_per_device inside pmap: {shape_val}", shape_val=local_data_slice_per_device.shape)
    
    result_slice = jnp.sum(local_data_slice_per_device, axis=1) * (device_idx_in_pmap + 1.0)
    return result_slice

# Áp dụng pmap một cách tường minh
parallel_computation = jax.pmap(_parallel_computation_implementation, axis_name='data_parallel_axis')


def main():
    print("--- JAX Multi-Device Check (using pmap in single process) ---")
    print(f"JAX Version: {jax.__version__}")
    print(f"jaxlib Version: {jax.lib.__version__}")

    try:
        print(f"KERAS_BACKEND (from env): {os.environ.get('KERAS_BACKEND')}")

        local_devices = jax.local_devices()
        print(f"Local JAX devices: {local_devices}")
        num_local_devices = jax.local_device_count()
        print(f"Number of local JAX devices: {num_local_devices}")

        if num_local_devices == 0:
            print("ERROR: No local JAX devices found. Cannot perform pmap test.")
            return
        if local_devices and "TPU" not in str(local_devices[0].platform).upper():
            print(f"WARNING: First local device is {local_devices[0].platform}, not detected as TPU. pmap will still run.")
        
        # Dữ liệu mẫu
        rows_per_device = 3 
        num_features = 4
        # Tổng số hàng là num_local_devices * rows_per_device
        total_rows = num_local_devices * rows_per_device 
        
        key = jax.random.PRNGKey(0)
        # Tạo dữ liệu phẳng ban đầu
        flat_data = jax.random.normal(key, (total_rows, num_features))
        print(f"Initial flat data shape: {flat_data.shape}")

        # Reshape dữ liệu để trục đầu tiên có kích thước bằng num_local_devices
        # Shape mới sẽ là (num_local_devices, rows_per_device, num_features)
        data_for_pmap = flat_data.reshape((num_local_devices, rows_per_device, num_features))
        print(f"Data reshaped for pmap (num_devices, rows_per_device, num_features): {data_for_pmap.shape}")

        result_sharded = parallel_computation(data_for_pmap) # Truyền dữ liệu đã reshape

        print(f"\nSharded result from pmap (one result slice per device):\n{result_sharded}")
        # Kết quả sẽ có shape (num_devices, rows_per_device) vì sum theo axis=1 của (rows_per_device, num_features)
        print(f"Shape of sharded result (num_devices, rows_per_device): {result_sharded.shape}") 
        
        print("Devices for each shard of the result (physical devices):")
        
        # In thông tin sharding của toàn bộ ShardedDeviceArray để debug
        if hasattr(result_sharded, 'sharding'):
            print(f"  Info: result_sharded.sharding: {result_sharded.sharding}")
            if hasattr(result_sharded.sharding, 'device_set'):
                 print(f"  Info: result_sharded.sharding.device_set: {result_sharded.sharding.device_set}")

        if hasattr(result_sharded, 'addressable_shards'):
            print(f"  Iterating through result_sharded.addressable_shards (type: {type(result_sharded.addressable_shards)})")
            for i, shard_item in enumerate(result_sharded.addressable_shards):
                print(f"    Shard {i} item type: {type(shard_item)}")
                actual_device = "Unknown"
                value_to_print_str = "N/A"

                # Thử các cách khác nhau để lấy device và data từ shard_item
                if hasattr(shard_item, 'devices') and callable(shard_item.devices): # Nếu shard_item là JAX array
                    device_set_for_shard = shard_item.devices()
                    actual_device = list(device_set_for_shard)[0] if device_set_for_shard else "Unknown (from .devices())"
                    value_to_print_str = str(shard_item[0]) if shard_item.size > 0 else "N/A (empty JAX array shard)"
                elif hasattr(shard_item, 'device'): # Nếu shard_item là một đối tượng Shard có thuộc tính .device
                    actual_device = shard_item.device
                    if hasattr(shard_item, 'data') and shard_item.data is not None: # Đối tượng Shard có thể có trường .data
                         value_to_print_str = str(shard_item.data[0]) if shard_item.data.size > 0 else "N/A (empty .data)"
                    elif hasattr(shard_item, 'is_fully_replicated') and shard_item.is_fully_replicated:
                         value_to_print_str = "Replicated shard data (check specific API)"
                    else: # Cố gắng coi shard_item như một array nếu không có .data
                        try:
                            # Kiểm tra xem shard_item có thể được đánh chỉ số và có thuộc tính size không
                            if hasattr(shard_item, '__getitem__') and hasattr(shard_item, 'size'):
                                value_to_print_str = str(shard_item[0]) if shard_item.size > 0 else "N/A (empty shard-like obj)"
                        except Exception as e_access:
                            # print(f"      DEBUG: Could not access shard_item like an array: {e_access}")
                            pass 
                else: 
                    print(f"    Could not determine device for shard {i} using common attributes.")
                
                print(f"    Shard {i} (value first element: {value_to_print_str}) is on actual device: {actual_device}")
        elif hasattr(result_sharded, 'shards') and callable(result_sharded.shards): # Một số API cũ hơn có thể dùng result_sharded.shards()
            print(f"  Iterating through result_sharded.shards() (type: {type(result_sharded.shards())})")
            # result_sharded.shards() có thể trả về list các JAX array
            for i, shard_array in enumerate(result_sharded.shards()):
                 print(f"    Shard {i} item type: {type(shard_array)}")
                 device_set_for_shard = shard_array.devices()
                 actual_device = list(device_set_for_shard)[0] if device_set_for_shard else "Unknown (from .shards().devices())"
                 value_to_print_str = str(shard_array[0]) if shard_array.size > 0 else "N/A (empty JAX array shard)"
                 print(f"    Shard {i} (value first element: {value_to_print_str}) is on actual device: {actual_device}")
        else:
            print("  Could not determine sharding information from result_sharded (neither .addressable_shards nor .shards found/usable).")
        
        print("\n--- Ghi chú về JAX Multi-Process thực sự ---")
        print("Để chạy JAX trên nhiều process (ví dụ: nhiều VM/node TPU):")
        print("1. Cần khởi tạo JAX distributed: `jax.distributed.initialize()` hoặc tương tự.")
        print("2. Mỗi process sẽ chạy cùng một script.")
        print("3. `jax.process_index()` và `jax.process_count()` sẽ cho biết ID và tổng số process.")
        print("4. Dữ liệu và model sharding (ví dụ: với `pjit` và `Mesh`) sẽ cần thiết để phân phối công việc.")
        print("5. Thường sử dụng các công cụ như SLURM, MPI, hoặc các script khởi chạy tùy chỉnh của cloud provider.")
        print("Script này chỉ mô phỏng multi-device trong một process bằng `pmap` trên các thiết bị cục bộ.")

        print("\nJAX multi-device (pmap) check completed.")

    except Exception as e:
        print(f"ERROR during JAX multi-device (pmap) check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
