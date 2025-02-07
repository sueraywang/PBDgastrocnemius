import time
import cProfile
import pstats
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, List
import numpy as np
from enum import Enum, auto
import threading
from collections import defaultdict
import line_profiler

class TimerType(Enum):
    CPU = auto()
    WALL = auto()
    GPU = auto()  # Placeholder for future GPU timing integration
    MEMORY = auto()  # For memory profiling
    
class ProfilingScope:
    def __init__(self, name: str, parent: Optional['ProfilingScope'] = None):
        self.name = name
        self.parent = parent
        self.children: Dict[str, 'ProfilingScope'] = {}
        self.timing_stats: Dict[TimerType, List[float]] = defaultdict(list)
        self.call_count = 0
        
    def add_timing(self, duration: float, timer_type: TimerType = TimerType.WALL):
        self.timing_stats[timer_type].append(duration)
        
    def get_stats(self, timer_type: TimerType = TimerType.WALL):
        times = self.timing_stats[timer_type]
        if not times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "calls": 0}
        
        return {
            "mean": np.mean(times),
            "std": np.std(times) if len(times) > 1 else 0,
            "min": np.min(times),
            "max": np.max(times),
            "calls": len(times)
        }

class Profiler:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Profiler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.root = ProfilingScope("root")
        self.current_scope = self.root
        self.enabled = True
        self.cprofile = cProfile.Profile()
        self.line_profiler = line_profiler.LineProfiler()
        self._scope_stack = []
        
    @contextmanager
    def profile_scope(self, name: str, timer_type: TimerType = TimerType.WALL):
        """Context manager for profiling a scope"""
        if not self.enabled:
            yield
            return
            
        if name not in self.current_scope.children:
            self.current_scope.children[name] = ProfilingScope(name, self.current_scope)
            
        previous_scope = self.current_scope
        self.current_scope = self.current_scope.children[name]
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.current_scope.add_timing(duration, timer_type)
            self.current_scope = previous_scope
            
    def profile_function(self, timer_type: TimerType = TimerType.WALL):
        """Decorator for profiling functions"""
        def decorator(func):
            # Add line profiling
            func = self.line_profiler(func)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_scope(func.__name__, timer_type):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling a block of code"""
        with self.profile_scope(name):
            yield
            
    def start_profiling(self):
        """Start profiling session"""
        self.enabled = True
        self.cprofile.enable()
        
    def stop_profiling(self):
        """Stop profiling session"""
        self.enabled = False
        self.cprofile.disable()
        
    def reset(self):
        """Reset all profiling data"""
        self.root = ProfilingScope("root")
        self.current_scope = self.root
        self.cprofile = cProfile.Profile()
        
    def get_stats(self, timer_type: TimerType = TimerType.WALL):
        """Get hierarchical profiling statistics"""
        def _get_scope_stats(scope: ProfilingScope, depth: int = 0):
            stats = {
                "name": scope.name,
                "stats": scope.get_stats(timer_type),
                "depth": depth
            }
            
            if scope.children:
                stats["children"] = [
                    _get_scope_stats(child, depth + 1)
                    for child in scope.children.values()
                ]
            
            return stats
            
        return _get_scope_stats(self.root)
    
    def print_stats(self, timer_type: TimerType = TimerType.WALL):
        """Print formatted profiling statistics"""
        def _print_scope(stats, indent=""):
            print(f"{indent}{stats['name']}:")
            s = stats['stats']
            print(f"{indent}  Mean: {s['mean']*1000:.3f}ms")
            print(f"{indent}  Std:  {s['std']*1000:.3f}ms")
            print(f"{indent}  Min:  {s['min']*1000:.3f}ms")
            print(f"{indent}  Max:  {s['max']*1000:.3f}ms")
            print(f"{indent}  Calls: {s['calls']}")
            
            if 'children' in stats:
                for child in stats['children']:
                    _print_scope(child, indent + "  ")
                    
        _print_scope(self.get_stats(timer_type))
        
    def print_line_stats(self):
        """Print line-by-line profiling statistics"""
        self.line_profiler.print_stats()
        
    def get_cprofile_stats(self):
        """Get cProfile statistics"""
        stats = pstats.Stats(self.cprofile)
        stats.strip_dirs()
        return stats