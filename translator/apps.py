import os
import sys
import threading
import time
import subprocess
from django.apps import AppConfig

class TranslatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'translator'

    def ready(self):
        """Triggers when the Django web service reloads or initializes."""
        # Django runserver runs two processes to facilitate hot-reloads.
        # This check ensures our background thread watcher only kicks off once!
        if os.environ.get('RUN_MAIN') == 'true':
            threading.Thread(target=self.start_dataset_monitor_loop, daemon=True).start()

    def start_dataset_monitor_loop(self):
        """A dedicated background thread tracking dataset file increments for auto-retraining."""
        print(">> Mudrā MLOps Daemon initialized and running in background...")
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
        TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, 'train_model.py') # Kept your custom name
        
        # === SMART INITALIZATION PASS ===
        # Count what already exists on disk right now so we don't train on old data
        last_known_counts = {}
        if os.path.exists(DATASET_DIR):
            initial_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
            for file in initial_files:
                if '_' in file:
                    prefix = file.split('_')[0]
                    last_known_counts[prefix] = last_known_counts.get(prefix, 0) + 1
                    
        print(f"[MLOPS DAEMON] Baseline folder tracking synchronized: {last_known_counts}")

        while True:
            try:
                time.sleep(5)  # Quietly evaluate disk state once every 5 seconds
                
                if not os.path.exists(DATASET_DIR):
                    continue

                all_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
                
                # Gather current file counts grouped by gesture word prefix
                current_counts = {}
                for file in all_files:
                    if '_' in file:
                        prefix = file.split('_')[0]
                        current_counts[prefix] = current_counts.get(prefix, 0) + 1

                # Evaluate changes against our disk baseline
                for prefix, current_count in current_counts.items():
                    baseline_count = last_known_counts.get(prefix, 0)
                    
                    # TRIGGER CONDITION: 
                    # 1. Must have crossed the 15-file minimum threshold overall.
                    # 2. The current count must be HIGHER than our baseline (a new file was just saved!)
                    if current_count >= 15 and current_count > baseline_count:
                        print(f"\n[MLOPS TRIGGER] New sequence detected for '{prefix}' ({current_count}/15 files). Initiating background training...")
                        
                        # Update our baseline instantly so it doesn't trigger again on the next loop iteration
                        last_known_counts[prefix] = current_count

                        if os.path.exists(TRAIN_SCRIPT_PATH):
                            log_file_path = os.path.join(BASE_DIR, 'mlops_train_debug.log')
                            log_file = open(log_file_path, 'w')

                            subprocess.Popen(
                                [sys.executable, TRAIN_SCRIPT_PATH],
                                stdout=log_file,
                                stderr=log_file
                            )
                        else:
                            print(f"[MLOPS ERROR] Background train_model.py missing at '{TRAIN_SCRIPT_PATH}'")
                            
                # Sync any newly introduced words into our baseline tracking maps safely
                for prefix, current_count in current_counts.items():
                    if prefix not in last_known_counts:
                        last_known_counts[prefix] = current_count
                            
            except Exception as e:
                print(f"[MLOPS DAEMON EXCEPTION] Error looping directory state: {e}")