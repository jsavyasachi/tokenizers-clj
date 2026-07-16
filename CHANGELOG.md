# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-16

### Added
- Enriched encode results with offsets, sequence ids, overflow encodings, and
  maximum-length status.
- Added configurable truncation, maximum length, stride, padding, special tokens,
  lowercasing, and tokenizer config loading.
- Added paired-sequence encoding, paired and configurable batch encoding, and batch
  decoding.
- Added reproducible HuggingFace Hub loading with revision, authentication, cache,
  and local-only or offline options.

## [0.1.3] - 2026-07-12

### Changed
- Migrated the source-of-truth build from Leiningen to deps.edn and tools.build.

## [0.1.2] - 2026-06-28

### Fixed
- Added a preflight check for macOS x86_64 JVMs before DJL loads the native tokenizer,
  with an actionable Apple Silicon arm64 JDK message.

## [0.1.1] - 2026-06-26

### Changed
- Relicensed from EPL 1.0 to EPL 2.0 for cross-repo consistency.
- Bumped outdated dependencies and added tag-triggered Clojars release workflow.
- Standardized badges, community health files, and license notice formatting.

## [0.1.0] - 2026-06-22

### Added
- Initial release.
- `from-file`, `from-pretrained`, `from-stream` constructors over DJL's
  `HuggingFaceTokenizer` (native Rust `tokenizers` via JNI).
- `encode` returning a Clojure map (`:ids :tokens :type-ids :word-ids
  :attention-mask :special-tokens-mask`), with `:add-special-tokens?` /
  `:with-overflowing-tokens?` options.
- `ids`, `tokens`, `count-tokens`, `decode`, and `batch-encode` helpers.
- Tokenizers are `Closeable`, so `with-open` frees the native handle.

[0.2.0]: https://github.com/jsavyasachi/tokenizers-clj/releases/tag/v0.2.0
[0.1.2]: https://github.com/jsavyasachi/tokenizers-clj/releases/tag/v0.1.2
[0.1.0]: https://github.com/jsavyasachi/tokenizers-clj/releases/tag/0.1.0
