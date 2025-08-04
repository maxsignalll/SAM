import time

class PerfCounter:
    """
    A singleton wrapper for time.perf_counter() to ensure the entire application
    uses a unified clock source when high-precision timing is needed.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerfCounter, cls).__new__(cls)
            # perf_counter is already a function, we can reference it directly
            cls._instance.perf_counter = time.perf_counter
        return cls._instance

    @classmethod
    def instance(cls):
        """Get the singleton instance of PerfCounter"""
        if cls._instance is None:
            cls() # Call __new__ to create instance
        return cls._instance

# Create an instance when module loads, making it immediately available
PerfCounter.instance() 