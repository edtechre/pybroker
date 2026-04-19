| Backend          | Wall time | Correct | Notes                                                     |
|------------------|----------:|:-------:|-----------------------------------------------------------|
| loky             |   0.900 s |   yes   | default; fork + memmap for NumPy                          |
| threading        |   0.614 s |   yes   | GIL-bound; relevant post-free-thread (PEP 703)            |
| multiprocessing  |   0.629 s |   yes   | legacy; loky supersedes                                   |
| ray              |   1.116 s |   yes   | head @ ray://ray-head:10001 (pybroker-validator img)      |
| dask             |   0.516 s |   yes   | scheduler+worker @ tcp://:8786 (see note on joblib patch) |
| spark            |   4.578 s |   yes   | master+worker @ spark://:7077 (joblibspark via Py4J)      |
