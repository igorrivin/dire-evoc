# X announcement

## Recommended standalone post

> Introducing dire-evoc: an open-source package connecting GPU dimensionality reduction
> with density clustering. It combines DiRe-RAPIDS + EVoC, estimates intrinsic dimension,
> and predicts labels for new data. Tested with RAPIDS 26.06:
> https://github.com/igorrivin/dire-evoc

272 raw characters; 257 characters with X’s standard URL weighting.

## Optional four-post thread

### Post 1

> 1/4 I’m releasing dire-evoc, an open-source Python package connecting DiRe-RAPIDS and EVoC.
> It builds a GPU layout, finds density-based clusters, estimates intrinsic dimension, and
> predicts labels for new data—all in one workflow.

230 characters.

### Post 2

> 2/4 Why combine them? DiRe can expose structure hidden in high-dimensional data; EVoC finds
> clusters in the resulting layout. dire-evoc supplies the bridge. Its intrinsic-dimension
> estimate reuses DiRe’s neighbor graph, avoiding a second search.

245 characters.

### Post 3

> 3/4 In one NVIDIA A10 sweep using matched data and base controls, dire-evoc improved
> clustering scores on OptDigits, PenDigits, SatImage, and Image Segmentation. It tied on
> several synthetic datasets; pure EVoC was slightly better on MNIST. Results vary by
> dataset.

265 characters.

### Post 4

> 4/4 The trade-off: dire-evoc is heavier because it creates a layout before clustering. It
> pays off when that layout exposes structure raw-space clustering misses. Now tested with
> RAPIDS 26.06. Code, examples, and benchmark details:
> https://github.com/igorrivin/dire-evoc

270 raw characters; 255 characters with X’s standard URL weighting.
