# jax_single_device_check.py
import jax
import jax.numpy as jnp
import os

def main():
    print("--- JAX Single Device Check ---")
    print(f"JAX Version: {jax.__version__}")
    print(f"jaxlib Version: {jax.lib.__version__}")

    try:
        print(f"KERAS_BACKEND (from env): {os.environ.get('KERAS_BACKEND')}")
        
        devices = jax.devices()
        print(f"Available JAX devices: {devices}")

        if not devices or "TPU" not in str(devices[0]).upper():
            print("ERROR: No TPU devices found or JAX is not configured for TPU.")
            return

        # Chọn thiết bị TPU đầu tiên
        tpu_device = devices[0]
        print(f"Using TPU device: {tpu_device}")

        # Dữ liệu mẫu
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 4))
        print(f"Input data (first 2x2):\n{x[:2,:2]}")

        # Hàm JIT đơn giản
        @jax.jit
        def simple_computation(data):
            return jnp.dot(data, data.T) + 2.0

        # Thực thi trên thiết bị TPU mặc định (hoặc thiết bị được JIT chọn)
        with jax.default_device(tpu_device): # Context để JIT trên device cụ thể nếu muốn
            result = simple_computation(x)
        
        print(f"Computation result (first 2x2) on {result.devices().pop()}:\n{result[:2,:2]}")
        print("JAX single device check completed successfully.")

    except Exception as e:
        print(f"ERROR during JAX single device check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
