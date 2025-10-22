from collections import defaultdict
from threading import Lock
from typing import Dict, Any

# Runtime state shared across modules
bot_active_per_chat = defaultdict(lambda: True)
bot_state_lock = Lock()
conversation_history: Dict[str, Dict[str, Any]] = {}
