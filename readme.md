# TPU Keras JAX Project

This project provides scripts to set up the environment, request TPU resources on Google Cloud,
and test JAX, Keras (with JAX backend) on TPUs.

## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd tpu_keras_jax_project
    ```

2.  **Run the setup script:**
    This script will update the system, install necessary packages, create a Python virtual environment,
    install Python libraries, and set up basic environment variables.
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```
    After running, you might need to reactivate the virtual environment if it's a new terminal session:
    ```bash
    source ~/my_env/bin/activate
    ```
    And you may need to re-export the environment variables if they were not added to `~/.bashrc` or similar.

## Requesting TPU Resources (Queued Resources)

The scripts in the `scripts/` directory help you submit requests to create TPU VMs via Queued Resources.
**NOTE:** Please replace `PROJECT_ID`, `ZONE`, `NODE_ID`, `QR_REQUEST_NAME` in these scripts with your values if needed.

*   **Create Spot TPU VM:**
    ```bash
    chmod +x scripts/submit_spot_tpu_qr.sh
    ./scripts/submit_spot_tpu_qr.sh
    ```

*   **Create On-Demand TPU VM:**
    ```bash
    chmod +x scripts/submit_ondemand_tpu_qr.sh
    ./scripts/submit_ondemand_tpu_qr.sh
    ```

    After submitting a request, you can check its status using:
    ```bash
    gcloud alpha compute tpus queued-resources list --zone <YOUR_ZONE> --project <YOUR_PROJECT_ID>
    gcloud alpha compute tpus queued-resources describe <YOUR_QR_REQUEST_NAME> --zone <YOUR_ZONE> --project <YOUR_PROJECT_ID>
    ```
    Once the Queued Resource is provisioned and the TPU VM is ready, you can SSH into the node.

## Running Test Scripts

After SSHing into the TPU VM, activating the virtual environment (`source ~/my_env/bin/activate`), and exporting the necessary environment variables (`PJRT_DEVICE`, `KERAS_BACKEND`, `XLA_PYTHON_CLIENT_MEM_FRACTION`):

*   **JAX single device check:**
    ```bash
    python3 scripts/jax_single_device_check.py
    ```

*   **JAX multi-device check (pmap):**
    ```bash
    python3 scripts/jax_multi_device_check.py
    ```

*   **Keras multi-device training check (pmap with JAX backend):**
    ```bash
    python3 scripts/keras_multi_device_training.py
    ```

## Important Environment Variables

Ensure these variables are set in your terminal session on the TPU VM before running Python scripts:

```bash
export PJRT_DEVICE=TPU
export KERAS_BACKEND=jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=1.00 # Or 0.9 depending on needs
# export CUDA_VISIBLE_DEVICES="-1" # If you have a GPU and want to hide it from JAX/TF
```
