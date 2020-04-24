# Patches

1. In `.venv/lib/python3.7/site-packages/pattern/text/__init__.py` change
```
            try:
 609                 yield line
 610             except StopIteration:
 611                 return
 ```
 Ref: https://stackoverflow.com/questions/51700960/runtimeerror-generator-raised-stopiteration-every-time-i-try-to-run-app
