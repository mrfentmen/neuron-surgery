from pathlib import Path
Path(__file__).resolve().with_name('STOP').write_text('User requested stop.\n')
print('Stop requested. Completed replies remain saved.')
