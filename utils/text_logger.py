# text_logger.py
import os
import json
import datetime
from typing import Any, Dict

class TextLogger:
    def __init__(self, log_dir: str, filename: str = "train_log.txt"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        # create file header
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"Run started: {datetime.datetime.now().isoformat()}\n")

    def write(self, msg: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    def log_config(self, cfg: Dict[str, Any], title: str = "CONFIG"):
        self.write(f"\n[{title}]")
        self.write(json.dumps(cfg, indent=2, sort_keys=True))

    def log_metrics(self, epoch: int, metrics: Dict[str, Any], prefix: str = "EPOCH"):
        # Make it stable + readable
        keys = sorted(metrics.keys())
        self.write(f"\n[{prefix} {epoch}]")
        for k in keys:
            v = metrics[k]
            if isinstance(v, float):
                self.write(f"{k}: {v:.6f}")
            else:
                self.write(f"{k}: {v}")

    def log_event(self, event: str):
        self.write(f"[EVENT] {event}")
        
    def log_dict(self, epoch: int, logging_dict: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\nEpoch {epoch}\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(logging_dict.items()):
                if isinstance(v, float):
                    f.write(f"{k}: {v:.6f}\n")
                else:
                    f.write(f"{k}: {v}\n")
