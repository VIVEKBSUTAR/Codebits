import datetime

class SystemLogger:
    def __init__(self, module_name: str):
        self.module_name = module_name
        
    def log(self, message: str):
        # Enforce [time] module_name: message format
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] {self.module_name}: {message}")
