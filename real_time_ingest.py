# real_time_ingest.py
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import os

# Directories
LOG_DIR = "logs"  # folder where new CSVs/logs appear
PROCESSED_LOG = "processed_logs.csv"

class LogHandler(FileSystemEventHandler):
    def __init__(self, process_func):
        self.process_func = process_func

    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            print(f"New log detected: {event.src_path}")
            try:
                df = pd.read_csv(event.src_path, low_memory=False)
                processed_df = self.process_func(df)  # call app's process_new_logs
                # Save to persistent CSV
                if os.path.exists(PROCESSED_LOG):
                    processed_df.to_csv(PROCESSED_LOG, mode='a', header=False, index=False)
                else:
                    processed_df.to_csv(PROCESSED_LOG, index=False)
                print(f"Processed and saved: {event.src_path}")
            except Exception as e:
                print(f"Error processing file {event.src_path}: {e}")

def watch_logs(process_func):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    observer = Observer()
    handler = LogHandler(process_func)
    observer.schedule(handler, LOG_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def start_watching_in_thread(process_func):
    """Run log watcher in a daemon thread"""
    thread = threading.Thread(target=watch_logs, args=(process_func,), daemon=True)
    thread.start()
