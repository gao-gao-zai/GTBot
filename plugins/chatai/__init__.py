from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import ttttttt
import config_manager
rag_config = config_manager.config_group_data.Retrieval_Augmented_Generation
if rag_config.enable:
    import NRAGmanager



import record
import main

