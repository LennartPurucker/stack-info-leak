# stack-info-leak

AutoGluon is a required dependency for using this library, along with s3fs:

```bash
# Requires Python 3.7, 3.8, or 3.9
git clone https://github.com/Innixma/stack-info-leak.git
pip install s3fs
pip install --pre autogluon  # or source install
```

Try it out with the examples:

```bash
python examples/run_santander.py
```
