#!/usr/bin/env python3
"""Independent, lightweight process/memory/thermal timeout guard; no model imports."""
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import json
import os
import signal
import time
import psutil

HERE = Path(__file__).resolve().parent
state = json.loads((HERE / 'status.json').read_text())
pid = state['pid']
process = psutil.Process(pid)
created = process.create_time()
launch = time.monotonic()
foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.objc_getClass.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p
address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
get_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
get_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(address)
instance = get_object(objc.objc_getClass(b'NSProcessInfo'), objc.sel_registerName(b'processInfo'))
selector = objc.sel_registerName(b'thermalState')


def save(value):
    value.update(updated_utc=datetime.now(timezone.utc).isoformat(), target_pid=pid, watchdog_pid=os.getpid())
    tmp = HERE / 'watchdog_status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(HERE / 'watchdog_status.json')


save({'state': 'monitoring'})
while True:
    try:
        if not process.is_running() or process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE:
            save({'state': 'target_exited'})
            break
        state = json.loads((HERE / 'status.json').read_text())
        if state['state'] in ('complete', 'stopped', 'failed') and (HERE / f'closed_{pid}.json').exists():
            save({'state': 'target_finished', 'runner_state': state['state']})
            break
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['updated_utc'])).total_seconds()
        reason = None
        if get_integer(instance, selector) >= 2:
            reason = 'thermal_pressure'
        elif psutil.virtual_memory().available < 3 * 1024**3:
            reason = 'low_available_memory'
        elif process.memory_info().rss > 12 * 1024**3:
            reason = 'process_rss_above_12GiB'
        elif state['state'] == 'generating' and age > 120:
            reason = 'generation_stalled'
        elif age > 600:
            reason = 'runner_stalled'
        elif time.monotonic() - launch > 8 * 3600 + 120:
            reason = 'wall_time_limit'
        if reason:
            save({'state': 'terminating', 'reason': reason})
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired:
                process.kill()
            save({'state': 'terminated', 'reason': reason})
            break
        time.sleep(5)
    except psutil.NoSuchProcess:
        save({'state': 'target_exited'})
        break
