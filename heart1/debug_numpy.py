import os
import sys
import traceback

print('CWD:', os.getcwd())
print('\nFiles in CWD:')
for name in sorted(os.listdir('.')):
    print(' -', name)

print('\n--- sys.path ---')
for p in sys.path:
    print(p)

print('\nAttempting to import numpy...')
try:
    import numpy as np
    print('numpy imported successfully')
    print('numpy.__file__ =', getattr(np, '__file__', None))
except Exception:
    traceback.print_exc()
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d.lower().startswith('numpy'):
                print('Found local dir that may shadow numpy:', os.path.join(root, d))
    sys.exit(1)
