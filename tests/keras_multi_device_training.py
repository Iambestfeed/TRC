# keras_multi_device_training.py
import os
os.environ['KERAS_BACKEND'] = 'jax'
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import keras
import jax
import jax.numpy as jnp
import numpy as np
import optax # Thư viện optimizer phổ biến cho JAX

# --- Định nghĩa các hàm sẽ được pmap một cách tường minh ---

def _init_opt_state_implementation(p_device_params):
    # Giả sử optimizer đã được định nghĩa trong phạm vi có thể truy cập
    # (sẽ được truyền vào hoặc định nghĩa toàn cục/trong main)
    # Trong trường hợp này, chúng ta sẽ định nghĩa optimizer trong main và
    # hàm này sẽ được gọi sau khi optimizer được tạo.
    # Để đơn giản, chúng ta có thể làm cho hàm này nhận optimizer làm đối số
    # hoặc dựa vào việc nó được định nghĩa trong main.
    # Vì optimizer.init chỉ phụ thuộc vào params, cách này vẫn ổn.
    return optimizer.init(p_device_params)


def _functional_model_apply_impl(f_params_list, f_x_batch):
    # f_params_list là list các jax.Array [kernel_dense1, bias_dense1, ...]
    idx = 0
    h = jnp.dot(f_x_batch, f_params_list[idx]) + f_params_list[idx+1]; idx += 2
    h = jax.nn.relu(h)
    h = jnp.dot(h, f_params_list[idx]) + f_params_list[idx+1]; idx += 2
    h = jax.nn.relu(h)
    logits = jnp.dot(h, f_params_list[idx]) + f_params_list[idx+1]
    return logits

def _compute_loss_implementation(current_params_on_device, x_batch_on_device, y_batch_on_device):
    logits = _functional_model_apply_impl(current_params_on_device, x_batch_on_device)
    preds_sigmoid = jax.nn.sigmoid(logits)
    loss = -jnp.mean(y_batch_on_device * jnp.log(preds_sigmoid + 1e-8) + \
                     (1 - y_batch_on_device) * jnp.log(1 - preds_sigmoid + 1e-8))
    return loss

def _train_step_implementation(params_on_device, opt_state_on_device, x_batch_on_device, y_batch_on_device):
    # optimizer cũng cần được truy cập ở đây
    def loss_for_grad(p):
        return _compute_loss_implementation(p, x_batch_on_device, y_batch_on_device)

    loss_value, grads_list = jax.value_and_grad(loss_for_grad)(params_on_device)
    
    grads_list = jax.lax.pmean(grads_list, axis_name='data_devices_axis') 
    loss_value = jax.lax.pmean(loss_value, axis_name='data_devices_axis')

    updates, new_opt_state = optimizer.update(grads_list, opt_state_on_device, params_on_device)
    new_params = optax.apply_updates(params_on_device, updates)
    
    return new_params, new_opt_state, loss_value

# Biến toàn cục tạm thời cho optimizer để các hàm trên có thể truy cập
# Đây không phải là cách lý tưởng nhất cho code lớn, nhưng giúp đơn giản hóa ví dụ này
# Trong code thực tế, bạn có thể truyền optimizer vào hoặc dùng class.
optimizer = None


def main():
    global optimizer # Khai báo để có thể gán giá trị cho optimizer toàn cục

    print("--- Keras Multi-Device Training (JAX Backend with Custom Loop & pmap) ---")
    print(f"Keras Version: {keras.__version__}")
    print(f"Keras Backend: {keras.backend.backend()}")
    print(f"JAX Version: {jax.__version__}")

    try:
        local_devices = jax.local_devices()
        num_devices = jax.local_device_count()
        print(f"Found {num_devices} local JAX devices: {local_devices}")

        if num_devices == 0:
            print("ERROR: No local JAX devices found.")
            return
        if "TPU" not in str(local_devices[0].platform).upper():
             print(f"WARNING: First local device is {local_devices[0].platform}, not detected as TPU. Training will proceed on available devices.")
        if num_devices < 1:
            print("ERROR: No JAX devices available for pmap.")
            return
        if num_devices < 2:
            print("INFO: Only one local device. Multi-device training will run, but won't demonstrate parallelism benefits across distinct devices.")

        print("Tạo dữ liệu huấn luyện mẫu...")
        batch_size_per_device = 32 
        num_samples = batch_size_per_device * num_devices
        input_dim = 10
        x_global_np = np.random.rand(num_samples, input_dim).astype(np.float32)
        y_global_np = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
        x_sharded_np = x_global_np.reshape((num_devices, batch_size_per_device, input_dim))
        y_sharded_np = y_global_np.reshape((num_devices, batch_size_per_device, 1))
        x_sharded = jnp.array(x_sharded_np)
        y_sharded = jnp.array(y_sharded_np)
        print(f"Data sharded for {num_devices} devices, each with shape: {x_sharded.shape[1:]}")

        print("Tạo model Keras...")
        input_layer = keras.Input(shape=(input_dim,), name="input_layer")
        x_layer_1 = keras.layers.Dense(64, activation="relu", name="dense_1")(input_layer)
        x_layer_2 = keras.layers.Dense(32, activation="relu", name="dense_2")(x_layer_1)
        output_layer = keras.layers.Dense(1, activation="sigmoid", name="output_layer")(x_layer_2)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.summary(print_fn=lambda s: print(f"  {s}"))

        print("Building model to initialize weights...")
        _ = model(x_sharded[0, :1])
        print("Model built.")
        
        params = [v.value for v in model.trainable_variables]
        if not params:
            print("ERROR: Model parameters are empty.")
            return
        print(f"Extracted {len(params)} sets of trainable parameters.")

        replicated_params = jax.device_put_replicated(params, local_devices)
        print(f"Parameters replicated across devices.")

        # Định nghĩa optimizer ở đây để _init_opt_state_implementation và _train_step_implementation có thể thấy
        optimizer = optax.adam(learning_rate=0.001)
        
        # Áp dụng pmap một cách tường minh cho init_opt_state
        # axis_name có thể không cần thiết ở đây nếu init không có collective op
        init_opt_state_on_device_pmapped = jax.pmap(_init_opt_state_implementation, axis_name='batch_axis_for_opt_init')
        opt_state = init_opt_state_on_device_pmapped(replicated_params)
        print(f"Optimizer state initialized and replicated.")

        # Áp dụng pmap một cách tường minh cho train_step
        train_step_pmapped = jax.pmap(_train_step_implementation, axis_name='data_devices_axis')

        num_epochs = 3
        print(f"Bắt đầu huấn luyện {num_epochs} epochs trên {num_devices} devices...")

        for epoch in range(num_epochs):
            replicated_params, opt_state, loss_sharded = train_step_pmapped(
                replicated_params, 
                opt_state, 
                x_sharded,
                y_sharded
            )
            avg_loss_epoch = jax.device_get(loss_sharded[0])
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss_epoch:.4f}")

        print("Huấn luyện đa thiết bị hoàn tất (sử dụng pmap).")
        
        final_params_host = jax.tree_util.tree_map(lambda x_device_array: jax.device_get(x_device_array[0]), replicated_params)
        print(f"Một phần params cuối cùng (kernel của layer đầu tiên, từ device 0):\n{final_params_host[0][:2,:2]}")

    except Exception as e:
        print(f"ERROR during Keras multi-device training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
