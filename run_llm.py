import sys
import subprocess
import torch
import json
import os
import time
import uuid
import threading
import queue
from src.llm.args import ARG

venv_python = sys.executable  # this gives the current venv's Python path
print(f"Using venv Python interpreter: {venv_python}")


# models = ['mistral', 'mistral-large', 'llama', 'llama-large', 'qwen', 'qwen-large', 'gemma', 'gemma-large']

models = ['qwen-large', 'mistral-large']
# models = ['llama-large', 'gemma-large']
# models = ['mistral', 'llama', 'qwen', 'gemma']
data_sources = ['data/topic_model_output/20ng', 'data/topic_model_output/agris', 'data/topic_model_output/tweets_ny']
ratings = ['number']


args = []

"""for data_source in data_sources:
    for model in models:
        for rating in ratings:
            args.append(ARG(model, data_source, rating, coverage=False, diversity=True))"""

"""for data_source in data_sources:
    for model in models:
        for rating in ratings:
            args.append(ARG(model, data_source, rating, coverage=False, diversity=False, coherence=True, repetitive=True, readability=True))"""
"""data_sources = ['data/document_topic_alignment']
for data_source in data_sources:
    for model in models:
        for rating in ratings:
            args.append(ARG(model, data_source, rating, coverage=True))
"""
for data_source in data_sources:
    for model in models:
        for rating in ratings:
            args.append(ARG(model, data_source, rating, clustering_complementarity=True))

NUM_GPUS = torch.cuda.device_count()
PROC_PER_GPU = 1
# --- Start: Changes for GPU Queue Management ---
available_gpus = queue.Queue()
gpu_lock = threading.Lock()

# Fill queue with available GPU ids
for gpu_id in range(NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        available_gpus.put(gpu_id)
# --- End: Changes for GPU Queue Management ---


def run_subprocess(arg, gpu_id, cur_num, length_of_args):
    temp_file = f"temp_args_{uuid.uuid4().hex}.json"
    with open(temp_file, "w") as f:
        json.dump(arg.__dict__, f)

    print(f"[INFO] Launching {arg} on GPU {gpu_id} ({cur_num}/{length_of_args})")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Start subprocess with real-time output capture
    p = subprocess.Popen(
        [venv_python, "src/llm/evaluation.py", temp_file],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True
    )

    # Print subprocess output in real-time
    def stream_output(proc, name):
        try:
            for line in proc.stdout:
                print(f"[{str(cur_num)} / {str(length_of_args)}] [{name}] {line}", end='')
        except Exception as e:
            print(f"Error streaming output for {name}: {e}")


    threading.Thread(target=stream_output, args=(p, str(arg)), daemon=True).start()

    # --- Start: Removed cleanup thread ---
    # The cleanup (returning GPU and removing file) will now happen in the main loop
    # --- End: Removed cleanup thread ---

    # Return the process and the temp file path for cleanup later
    return p, temp_file

length_of_args = len(args)
cur_num = 0
# Simple GPU job scheduler
running = [] # Store tuples of (process, gpu_id, temp_file_path)
while args or running:
    # --- Start: Updated Job Completion Check ---
    # Check for finished jobs and return GPUs to the queue
    new_running = []
    for p, g, temp_file in running:
        if p.poll() is None:
            # Process is still running
            new_running.append((p, g, temp_file))
        else:
            # Process finished
            print(f"[INFO] Finished {p.args[-1].split('_')[-1].split('.')[0]} (originally {temp_file}) on GPU {g}") # Use temp_file for context if needed
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"[WARN] Could not remove temp file {temp_file}: {e}")
            # Return GPU to the queue
            with gpu_lock:
                available_gpus.put(g)
                print(f"[DEBUG] Returned GPU {g} to queue. Queue size: {available_gpus.qsize()}") # Optional debug print
    running = new_running
    # --- End: Updated Job Completion Check ---

    # --- Start: Updated Job Scheduling ---
    # Schedule new jobs if GPUs are available in the queue
    while args and not available_gpus.empty():
        # Get an available GPU from the queue
        with gpu_lock:
            # Check again inside the lock in case of race condition
            if available_gpus.empty():
                break
            gpu_id = available_gpus.get()
            print(f"[DEBUG] Got GPU {gpu_id} from queue. Queue size: {available_gpus.qsize()}") # Optional debug print


        cur_num += 1
        arg = args.pop(0)
        try:
            p, temp_file = run_subprocess(arg, gpu_id, cur_num, length_of_args)
            running.append((p, gpu_id, temp_file))
        except Exception as e:
            print(f"[ERROR] Failed to launch {arg} on GPU {gpu_id}: {e}")
            # Return GPU to the queue if launch failed
            with gpu_lock:
                available_gpus.put(gpu_id)
                print(f"[DEBUG] Returned GPU {gpu_id} to queue after launch failure. Queue size: {available_gpus.qsize()}") # Optional debug print

    # --- End: Updated Job Scheduling ---

    time.sleep(1)  # avoid busy-wait

print("[INFO] All tasks completed.")

