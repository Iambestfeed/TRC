# keras_single_device_training.py
import os
os.environ['KERAS_BACKEND'] = 'jax' # Đặt backend TRƯỚC khi import Keras
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # Cố gắng ẩn GPU nếu có

import keras
import jax
import jax.numpy as jnp # Keras JAX backend sẽ dùng jax.numpy
import numpy as np # Có thể dùng numpy cho dữ liệu ban đầu

def main():
    print("--- Keras Single Device Training (JAX Backend) ---")
    print(f"Keras Version: {keras.__version__}")
    print(f"Keras Backend: {keras.backend.backend()}") # Phải là 'jax'
    print(f"JAX Version: {jax.__version__}")

    try:
        tpu_devices = [d for d in jax.devices() if "TPU" in d.platform.upper()]
        if not tpu_devices:
            print("ERROR: No TPU devices found for JAX.")
            return
        
        # Keras với JAX backend thường sẽ tự động đặt các hoạt động lên TPU nếu có sẵn.
        # Bạn có thể dùng jax.default_device(tpu_devices[0]) nếu muốn chắc chắn.
        print(f"JAX will attempt to use available TPUs: {tpu_devices}")

        print("Tạo dữ liệu huấn luyện mẫu...")
        # Sử dụng numpy để tạo dữ liệu, sau đó Keras/JAX sẽ xử lý
        num_samples = 1000
        input_dim = 10
        # Dữ liệu huấn luyện ngẫu nhiên
        x_train_np = np.random.rand(num_samples, input_dim).astype(np.float32)
        y_train_np = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

        # Chuyển sang JAX arrays nếu muốn (Keras cũng có thể xử lý NumPy arrays)
        # x_train = jnp.array(x_train_np)
        # y_train = jnp.array(y_train_np)
        # Sử dụng NumPy arrays trực tiếp thường tiện hơn cho model.fit
        x_train = x_train_np
        y_train = y_train_np


        print("Tạo model Keras...")
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid") # Bài toán phân loại nhị phân
        ])
        model.summary()

        print("Biên dịch model...")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.Accuracy()])

        print("Bắt đầu huấn luyện model...")
        history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)

        print("Huấn luyện hoàn tất.")
        final_loss = history.history.get('loss', [None])[-1]
        final_accuracy_key = next((key for key in history.history if 'accuracy' in key.lower()), None)
        final_accuracy = history.history.get(final_accuracy_key, [None])[-1]

        if final_loss is not None:
            print(f"Final training loss: {final_loss:.4f}")
        if final_accuracy is not None:
            print(f"Final training accuracy ({final_accuracy_key}): {final_accuracy:.4f}")

        # Kiểm tra thiết bị của một tham số
        if model.weights:
            first_weight_var = model.weights[0]
            # .value để lấy jax.Array từ Variable của Keras
            first_weight_devices = first_weight_var.value.devices()
            print(f"Device of the first model weight: {list(first_weight_devices)[0] if first_weight_devices else 'Unknown'}")
        
        print("Keras single device training check completed successfully.")

    except Exception as e:
        print(f"ERROR during Keras single device training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
