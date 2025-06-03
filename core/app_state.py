from core.database import list_available_databases

class AppState:
    """Manages the application state for the multi-mode interface"""
    def __init__(self):
        self.mode = "landing"  # "landing", "building", "searching"
        self.available_databases = []
        self.active_database = None
        print(f"[STATUS] Initializing App State")
        self.update_available_databases(verbose=False)

    def update_available_databases(self, verbose=True):
        """Update the list of available databases"""
        if verbose:
            print(f"[STATUS] Refreshing database list")

        self.available_databases = list_available_databases() # This function is now imported

        if verbose:
            print(f"[STATUS] Found {len(self.available_databases)} databases")

        return self.available_databases
