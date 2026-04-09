# Changelog

## [0.6.1](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.6.0...v0.6.1) (2026-04-09)


### Performance Improvements

* use borrowed split-chunk keys during training aggregation ([#155](https://github.com/zhixiangli/fast-bpe-rs/issues/155)) ([5443b8f](https://github.com/zhixiangli/fast-bpe-rs/commit/5443b8fc676417f399838eed5230df783836a0f6))

## [0.6.0](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.5.3...v0.6.0) (2026-04-07)


### Features

* add optional arrow zero-copy training support ([#147](https://github.com/zhixiangli/fast-bpe-rs/issues/147)) ([617dffe](https://github.com/zhixiangli/fast-bpe-rs/commit/617dffed41ab3cad18a5a3105c870b1ff24aefe2))


### Performance Improvements

* **python:** train wikitext via pyarrow pipeline ([#149](https://github.com/zhixiangli/fast-bpe-rs/issues/149)) ([90a317b](https://github.com/zhixiangli/fast-bpe-rs/commit/90a317bd7794cdae1930a0f0878861c4522f703f))

## [0.5.3](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.5.2...v0.5.3) (2026-04-07)


### Performance Improvements

* **python:** batch train docs conversion across PyO3 boundary ([#143](https://github.com/zhixiangli/fast-bpe-rs/issues/143)) ([bbe904c](https://github.com/zhixiangli/fast-bpe-rs/commit/bbe904ccb8f7a0d2b72cf088003b8db6b9bf2ac1))

## [0.5.2](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.5.1...v0.5.2) (2026-04-05)


### Performance Improvements

* use mimalloc as global allocator ([#140](https://github.com/zhixiangli/fast-bpe-rs/issues/140)) ([1935af9](https://github.com/zhixiangli/fast-bpe-rs/commit/1935af974f9ed406b5d7a51b0ae8f5cbaec5cda8))

## [0.5.1](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.5.0...v0.5.1) (2026-04-05)


### Performance Improvements

* reduce Python-to-Rust training overhead in PyO3 wrapper ([#138](https://github.com/zhixiangli/fast-bpe-rs/issues/138)) ([3b89b18](https://github.com/zhixiangli/fast-bpe-rs/commit/3b89b18bd2b4f04d4e3611dd5215ca4cba279671))

## [0.5.0](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.5...v0.5.0) (2026-04-05)


### Features

* add default fancy-regex split pattern ([#84](https://github.com/zhixiangli/fast-bpe-rs/issues/84)) ([ddf4d60](https://github.com/zhixiangli/fast-bpe-rs/commit/ddf4d60f74af5f3409aabe95660fd93e64f35399))
* add selectable WikiText dataset config and report filtered rows in training script ([#121](https://github.com/zhixiangli/fast-bpe-rs/issues/121)) ([5aa3abd](https://github.com/zhixiangli/fast-bpe-rs/commit/5aa3abd9d11333d4544873e5cb437718d3a4a7d2))
* **cli:** add flag to load wikitext-2 dataset ([#92](https://github.com/zhixiangli/fast-bpe-rs/issues/92)) ([e629ebd](https://github.com/zhixiangli/fast-bpe-rs/commit/e629ebd00568650dab9222caabb2b34ed42c4709))


### Bug Fixes

* **ci:** set repo for release uploads ([#61](https://github.com/zhixiangli/fast-bpe-rs/issues/61)) ([9057c15](https://github.com/zhixiangli/fast-bpe-rs/commit/9057c153e847f01bce4d18dc6e3940f6b06e3bcd))
* handle invalid split regex in python bindings ([#7](https://github.com/zhixiangli/fast-bpe-rs/issues/7)) ([151d0c1](https://github.com/zhixiangli/fast-bpe-rs/commit/151d0c17c1f40d127cb74dd7de69fe2324603a4b))
* normalize python unicode decode errors ([#9](https://github.com/zhixiangli/fast-bpe-rs/issues/9)) ([0d4c063](https://github.com/zhixiangli/fast-bpe-rs/commit/0d4c063080ade3919c12823d32529371f4da240b))
* prevent duplicate release PRs by removing draft release config ([#42](https://github.com/zhixiangli/fast-bpe-rs/issues/42)) ([b5c0987](https://github.com/zhixiangli/fast-bpe-rs/commit/b5c098713c2d156687ad800ba61bd53e1342a489))
* resolve merge conflict in training benchmark binary ([#91](https://github.com/zhixiangli/fast-bpe-rs/issues/91)) ([bf70ad5](https://github.com/zhixiangli/fast-bpe-rs/commit/bf70ad5410d3e0ee376b5c769d19ed0a4b36bfe2))
* restore full release flow for manual reruns ([#58](https://github.com/zhixiangli/fast-bpe-rs/issues/58)) ([059bbe2](https://github.com/zhixiangli/fast-bpe-rs/commit/059bbe29eb20aa3448ab731915be8c007b9ea1f5))
* route WikiText training logs to stdout ([#135](https://github.com/zhixiangli/fast-bpe-rs/issues/135)) ([06d3bee](https://github.com/zhixiangli/fast-bpe-rs/commit/06d3bee239272a34b193d500dbd38933d1c44364))
* set release to draft before uploading assets on retry ([#37](https://github.com/zhixiangli/fast-bpe-rs/issues/37)) ([3d45a36](https://github.com/zhixiangli/fast-bpe-rs/commit/3d45a36790160494e6fe951ab6dde99bcbf18927))


### Performance Improvements

* batch merge-candidate bucket updates per selected merge ([#133](https://github.com/zhixiangli/fast-bpe-rs/issues/133)) ([878d714](https://github.com/zhixiangli/fast-bpe-rs/commit/878d714dcec297d82eaa42ed4763dce9d08e4733))
* **ci:** speed up package checks job ([#94](https://github.com/zhixiangli/fast-bpe-rs/issues/94)) ([1a4b522](https://github.com/zhixiangli/fast-bpe-rs/commit/1a4b5222b224fa7925060e7c0b249c9f11065543))
* compile worker-local regexes during training chain build ([#106](https://github.com/zhixiangli/fast-bpe-rs/issues/106)) ([62fad02](https://github.com/zhixiangli/fast-bpe-rs/commit/62fad0209a3d083198ae9c7f7f14ff99b176ece3))
* expand memory benchmark stats output ([#95](https://github.com/zhixiangli/fast-bpe-rs/issues/95)) ([36893da](https://github.com/zhixiangli/fast-bpe-rs/commit/36893dad16111da4918217e048a13054bb1b3205))
* introduce `TrainingChunk` alias and widen SmallVec capacity for training data ([#108](https://github.com/zhixiangli/fast-bpe-rs/issues/108)) ([012343b](https://github.com/zhixiangli/fast-bpe-rs/commit/012343b1089383eef02bede96e135f9694433914))
* log BPE training duration at info level ([#126](https://github.com/zhixiangli/fast-bpe-rs/issues/126)) ([d8ad9b6](https://github.com/zhixiangli/fast-bpe-rs/commit/d8ad9b62cba655cb48fc6bb4c054161e996c2b45))
* make ensure_bucket logarithmic with ordered counts ([#127](https://github.com/zhixiangli/fast-bpe-rs/issues/127)) ([839f059](https://github.com/zhixiangli/fast-bpe-rs/commit/839f059113e41dce8de75e226624d13cf04630db))
* optimize training chunk frequency map lookups ([#110](https://github.com/zhixiangli/fast-bpe-rs/issues/110)) ([bc711e4](https://github.com/zhixiangli/fast-bpe-rs/commit/bc711e4daca56d91d3a042d4cd7b046129b2695c))
* run training twice with speed and memory profiling ([#89](https://github.com/zhixiangli/fast-bpe-rs/issues/89)) ([ef1ff22](https://github.com/zhixiangli/fast-bpe-rs/commit/ef1ff22e329685b535ea40ac2d7992a1760fe01d))
* switch training chunk map to FxHasher ([#130](https://github.com/zhixiangli/fast-bpe-rs/issues/130)) ([d1df0bd](https://github.com/zhixiangli/fast-bpe-rs/commit/d1df0bdf35eb54dfa1d2e1860e17c11e35b5d245))
* use FxHashMap for BPE state and reserve train capacities ([#124](https://github.com/zhixiangli/fast-bpe-rs/issues/124)) ([50602db](https://github.com/zhixiangli/fast-bpe-rs/commit/50602db8a82ccab6d249139461cbc98f59fac54a))
* use hashmap buckets for token-pair frequency index ([#115](https://github.com/zhixiangli/fast-bpe-rs/issues/115)) ([85cdd5d](https://github.com/zhixiangli/fast-bpe-rs/commit/85cdd5dced887117fb4666149e3e7c3d5c360722))

## [0.4.5](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.4...v0.4.5) (2026-04-05)


### Bug Fixes

* route WikiText training logs to stdout ([#135](https://github.com/zhixiangli/fast-bpe-rs/issues/135)) ([06d3bee](https://github.com/zhixiangli/fast-bpe-rs/commit/06d3bee239272a34b193d500dbd38933d1c44364))

## [0.4.4](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.3...v0.4.4) (2026-04-05)


### Performance Improvements

* batch merge-candidate bucket updates per selected merge ([#133](https://github.com/zhixiangli/fast-bpe-rs/issues/133)) ([878d714](https://github.com/zhixiangli/fast-bpe-rs/commit/878d714dcec297d82eaa42ed4763dce9d08e4733))

## [0.4.3](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.2...v0.4.3) (2026-04-05)


### Performance Improvements

* switch training chunk map to FxHasher ([#130](https://github.com/zhixiangli/fast-bpe-rs/issues/130)) ([d1df0bd](https://github.com/zhixiangli/fast-bpe-rs/commit/d1df0bdf35eb54dfa1d2e1860e17c11e35b5d245))

## [0.4.2](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.1...v0.4.2) (2026-04-05)


### Performance Improvements

* log BPE training duration at info level ([#126](https://github.com/zhixiangli/fast-bpe-rs/issues/126)) ([d8ad9b6](https://github.com/zhixiangli/fast-bpe-rs/commit/d8ad9b62cba655cb48fc6bb4c054161e996c2b45))
* make ensure_bucket logarithmic with ordered counts ([#127](https://github.com/zhixiangli/fast-bpe-rs/issues/127)) ([839f059](https://github.com/zhixiangli/fast-bpe-rs/commit/839f059113e41dce8de75e226624d13cf04630db))

## [0.4.1](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.4.0...v0.4.1) (2026-04-05)


### Performance Improvements

* use FxHashMap for BPE state and reserve train capacities ([#124](https://github.com/zhixiangli/fast-bpe-rs/issues/124)) ([50602db](https://github.com/zhixiangli/fast-bpe-rs/commit/50602db8a82ccab6d249139461cbc98f59fac54a))

## [0.4.0](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.3.4...v0.4.0) (2026-04-04)


### Features

* add selectable WikiText dataset config and report filtered rows in training script ([#121](https://github.com/zhixiangli/fast-bpe-rs/issues/121)) ([5aa3abd](https://github.com/zhixiangli/fast-bpe-rs/commit/5aa3abd9d11333d4544873e5cb437718d3a4a7d2))

## [0.3.4](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.3.3...v0.3.4) (2026-04-04)


### Performance Improvements

* use hashmap buckets for token-pair frequency index ([#115](https://github.com/zhixiangli/fast-bpe-rs/issues/115)) ([85cdd5d](https://github.com/zhixiangli/fast-bpe-rs/commit/85cdd5dced887117fb4666149e3e7c3d5c360722))

## [0.3.3](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.3.2...v0.3.3) (2026-04-04)


### Performance Improvements

* optimize training chunk frequency map lookups ([#110](https://github.com/zhixiangli/fast-bpe-rs/issues/110)) ([bc711e4](https://github.com/zhixiangli/fast-bpe-rs/commit/bc711e4daca56d91d3a042d4cd7b046129b2695c))

## [0.3.2](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.3.1...v0.3.2) (2026-04-04)


### Performance Improvements

* introduce `TrainingChunk` alias and widen SmallVec capacity for training data ([#108](https://github.com/zhixiangli/fast-bpe-rs/issues/108)) ([012343b](https://github.com/zhixiangli/fast-bpe-rs/commit/012343b1089383eef02bede96e135f9694433914))

## [0.3.1](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.3.0...v0.3.1) (2026-04-04)


### Performance Improvements

* compile worker-local regexes during training chain build ([#106](https://github.com/zhixiangli/fast-bpe-rs/issues/106)) ([62fad02](https://github.com/zhixiangli/fast-bpe-rs/commit/62fad0209a3d083198ae9c7f7f14ff99b176ece3))

## [0.3.0](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.2.1...v0.3.0) (2026-04-03)


### Features

* **cli:** add flag to load wikitext-2 dataset ([#92](https://github.com/zhixiangli/fast-bpe-rs/issues/92)) ([e629ebd](https://github.com/zhixiangli/fast-bpe-rs/commit/e629ebd00568650dab9222caabb2b34ed42c4709))


### Performance Improvements

* **ci:** speed up package checks job ([#94](https://github.com/zhixiangli/fast-bpe-rs/issues/94)) ([1a4b522](https://github.com/zhixiangli/fast-bpe-rs/commit/1a4b5222b224fa7925060e7c0b249c9f11065543))
* expand memory benchmark stats output ([#95](https://github.com/zhixiangli/fast-bpe-rs/issues/95)) ([36893da](https://github.com/zhixiangli/fast-bpe-rs/commit/36893dad16111da4918217e048a13054bb1b3205))

## [0.2.1](https://github.com/zhixiangli/fast-bpe-rs/compare/v0.2.0...v0.2.1) (2026-04-01)


### Bug Fixes

* resolve merge conflict in training benchmark binary ([#91](https://github.com/zhixiangli/fast-bpe-rs/issues/91)) ([bf70ad5](https://github.com/zhixiangli/fast-bpe-rs/commit/bf70ad5410d3e0ee376b5c769d19ed0a4b36bfe2))


### Performance Improvements

* run training twice with speed and memory profiling ([#89](https://github.com/zhixiangli/fast-bpe-rs/issues/89)) ([ef1ff22](https://github.com/zhixiangli/fast-bpe-rs/commit/ef1ff22e329685b535ea40ac2d7992a1760fe01d))
