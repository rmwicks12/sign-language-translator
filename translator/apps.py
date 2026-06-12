import os
import sys
import threading
import time
import subprocess
from django.apps import AppConfig

def get_next_training_threshold(current_baseline):
    """
    Calculates the next progressive milestone for model training.
    Prevents model thrashing by batching new entries exponentially.
    """
    if current_baseline < 15:
        return 15
    elif current_baseline < 20:
        return 20
    elif current_baseline < 30:
        return 30
    elif current_baseline < 50:
        return 50
    else:
        # After 50, batch train every 25 new samples to keep the model sharp
        return current_baseline + 25

class TranslatorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'translator'

    def ready(self):
        """Triggers when the Django web service reloads or initializes."""
        if os.environ.get('RUN_MAIN') == 'true':
            
            def delayed_daemon_launch():
                time.sleep(2.0)
                self.start_dataset_monitor_loop()

            startup_worker = threading.Thread(target=delayed_daemon_launch, daemon=True)
            startup_worker.start()

    def start_dataset_monitor_loop(self):
        """A dedicated background thread tracking dataset file increments for auto-retraining."""
        print(">> Mudrā MLOps Daemon initialized and running in background...")
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
        TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, 'train_model.py')
        
        # === SMART INITIALIZATION PASS ===
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
                time.sleep(5)
                
                if not os.path.exists(DATASET_DIR):
                    continue

                all_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
                
                current_counts = {}
                for file in all_files:
                    if '_' in file:
                        prefix = file.split('_')[0]
                        current_counts[prefix] = current_counts.get(prefix, 0) + 1

                # Evaluate changes against our progressive milestones
                for prefix, current_count in current_counts.items():
                    baseline_count = last_known_counts.get(prefix, 0)
                    
                    # Calculate exactly what number we need to hit to trigger the next train
                    target_threshold = get_next_training_threshold(baseline_count)
                    
                    # TRIGGER CONDITION: Have we crossed the target milestone?
                    if current_count >= target_threshold:
                        print(f"\n[MLOPS TRIGGER] Milestone reached for '{prefix}' ({current_count} files). Target was {target_threshold}. Initiating background training...")
                        
                        # Update baseline instantly to the new count so it waits for the NEXT threshold
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
                            
                # Sync any newly introduced words into our tracking maps safely
                for prefix, current_count in current_counts.items():
                    if prefix not in last_known_counts:
                        last_known_counts[prefix] = current_count
                            
            except Exception as e:
                print(f"[MLOPS DAEMON EXCEPTION] Error looping directory state: {e}")