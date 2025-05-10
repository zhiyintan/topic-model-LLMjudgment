from dataclasses import dataclass
import sys
import subprocess
import torch
import os
import time
import threading
import queue

venv_python = sys.executable  # this gives the current venv's Python path
print(f"Using venv Python interpreter: {venv_python}")

@dataclass
class ARG:
    model: str
    data_source: str

    def __repr__(self):
        return f'{self.model}+{self.data_source}'

models = ['llama', 'gemma', 'mistral', 'mistral-large', 'qwen', 'llama-large', 'qwen-large', 'gemma-large']
data_sources = ['random_100_topic_words_20ng.csv']
data_sources += ['random_100_topic_words_tweetsnyr.csv']
data_sources += ['random_100_topic_words_agris.csv']


args = []
for data_source in data_sources:
    for model in models:
        args.append(ARG(model, data_source))


NUM_GPUS = torch.cuda.device_count()
PROC_PER_GPU = 1
available_gpus = queue.Queue()
gpu_lock = threading.Lock()

# Fill queue with available GPU ids
for gpu_id in range(NUM_GPUS):
    for _ in range(PROC_PER_GPU):
        available_gpus.put(gpu_id)


def run_subprocess(arg, gpu_id):
    arg.cuda = str(gpu_id)
    print(f"[INFO] Launching {arg} on GPU {gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = arg.cuda

    # Start subprocess with real-time output capture
    p = subprocess.Popen(
        [venv_python, "adversial_test/adversial_test.py", "--llm_model", arg.model, "--filename", arg.data_source],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True
    )

    # Print subprocess output in real-time
    def stream_output(proc, name):
        for line in proc.stdout:
            print(f"[{name}] {line}", end='')

    threading.Thread(target=stream_output, args=(p, str(arg)), daemon=True).start()

    return p


# Simple GPU job scheduler
running = []
while args or running:
    # Remove finished jobs
    new_running = []
    for p, g in running:
        if p.poll() is None:
            new_running.append((p,g))
        else:
            with gpu_lock:
                available_gpus.put(g)
    running = new_running

    # Schedule new jobs
    while args and not available_gpus.empty():
        with gpu_lock:
            gpu_id = available_gpus.get()
        arg = args.pop(0)
        p = run_subprocess(arg, gpu_id)
        running.append((p, gpu_id))
    
    time.sleep(1)  # avoid busy-wait
