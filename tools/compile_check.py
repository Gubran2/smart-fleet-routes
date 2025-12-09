import py_compile
p = r'c:\Users\gubra\OneDrive\Skrivbord\ping pong\smart drive cars\smart-fleet-routes\app.py'
try:
    py_compile.compile(p, doraise=True)
    print('COMPILE_OK')
except Exception as e:
    print('COMPILE_ERROR', e)
    raise
