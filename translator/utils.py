import os

def get_dynamic_actions():
    """
    Scans the local dataset directory, parses file prefixes, 
    and returns a sorted list of unique registered gesture actions.
    """
    # Navigate cleanly to your root project workspace folder
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    
    # Fallback default if the folder hasn't been initialized yet
    if not os.path.exists(DATASET_DIR):
        return ['awaiting_data']
        
    # Read all flat JSON documents
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    
    # Extract the word prefix (e.g., "four" from "four_1781175920.json")
    actions = set()
    for file in files:
        if '_' in file:
            prefix = file.split('_')[0]
            actions.add(prefix)
            
    # If the folder is empty, fallback to default model categories
    if not actions:
        return ['awaiting_data']
        
    # Return as a cleanly sorted list for uniform index classification matching
    return sorted(list(actions))