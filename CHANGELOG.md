# Changelog

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
