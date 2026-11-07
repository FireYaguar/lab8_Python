| Query | Docs matched | Postings scored | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |
|---|---|---|---|---|---|
| common term: `the` | 552 | 552 | 0.650 | 0.051 | 12.7× |
| rare term: `winsound` | 5 | 5 | 0.007 | 0.016 | 0.4× |
| 3-term query (OR): `python OR async OR queue` | 475 | 568 | 0.585 | 0.079 | 7.4× |
| 3-term query (AND): `python async queue` | 13 | 568 | 0.056 | 0.064 | 0.9× |
