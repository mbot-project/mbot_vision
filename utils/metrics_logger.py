import os
import csv
import time
import psutil
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir='logs'):
        # Create logs directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'metrics_{timestamp}.csv')
        
        # Initialize CSV file with headers
        self.headers = ['elapsed_time_sec', 'fps', 'cpu_percent', 'memory_percent', 'memory_used_mb']
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        # Initialize process for getting CPU usage of current process
        self.process = psutil.Process(os.getpid())
        
        # Record start time
        self.start_time = time.time()
    
    def log_metrics(self, fps):
        """
        Log current metrics to CSV file with elapsed time in seconds
        """
        elapsed_time = round(time.time() - self.start_time, 2)
        cpu_percent = self.process.cpu_percent()
        memory_percent = self.process.memory_percent()
        memory_used = self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        metrics = [
            elapsed_time,
            round(fps, 2),
            round(cpu_percent, 2),
            round(memory_percent, 2),
            round(memory_used, 2)
        ]
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics) 