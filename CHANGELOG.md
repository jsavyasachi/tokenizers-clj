# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.11.0] - 2026-08-30

### Changed

- **BREAKING:** Encode offsets and span lookups now use UTF-16 indexes into the
  original source strings, suitable for direct use with `subs`.

### Security

- Hub downloads no longer forward bearer tokens across redirect origins and
  now enforce a bounded, manually-followed redirect chain.

### Fixed

- Hub cache lock entries are released after use without evicting an entry that
  another thread can still acquire or hold.

### Documentation

- Documented tokenizer lifecycle requirements and the upstream DJL 0.36.0
  exception-path native `Encoding` leak.

## [0.10.0] - 2026-08-28

### Added

- Hub cache downloads now enforce a maximum size, verify supplied SHA-256
  checksums, use atomic writes with cross-process cache locks, and clean up
  interrupted temporary files while preserving legacy and offline caches.

## [0.9.0] - 2026-08-27

### Added

- Native batch `batch-ids` and `batch-tokens` fast paths that avoid constructing
  full encode result maps.
- Batched pretokenized encoding with native word IDs and per-item overflow data.
- Public tokenizer configuration metadata exposes model limits, special tokens,
  tokenizer class, and normalization flags.
- CI now exercises Linux, macOS arm64, and Windows native runtime loading on
  JDK 17 and 21.
- README now includes GPT, Llama, Qwen, offline deployment, and NDList batch
  inference examples with explicit special-token and attention-mask guidance.

## [0.8.0] - 2026-08-27

### Added

- Token-budget text splitting and truncation helpers with native token IDs,
  original-string character offsets, and explicit overflow metadata.

## [0.7.0] - 2026-08-27

### Added

- HuggingFace Hub loading now caches and applies `tokenizer_config.json` metadata
  alongside `tokenizer.json`, including model max length and special tokens.

## [0.6.0] - 2026-08-27

### Added

- Public `tokenize` helper for getting token strings directly from a tokenizer
  without constructing a full encode result.

## [0.5.0] - 2026-08-17

### Fixed

- Hub tokenizer downloads apply connect and request timeouts instead of
  blocking forever on a stalled connection.

### Added

- `:download-timeout-ms` option for `from-pretrained` (30000 ms default).

## [0.4.0] - 2026-07-17

### Added
- Native batch token counting with `batch-count-tokens`.
- Encode-result span helpers: `token->chars`, `token->word`, `char->token`, and
  `word->tokens`.

### Changed
- `ids`, `tokens`, and `count-tokens` now read native arrays directly, avoiding
  full encode-map allocation with unchanged behavior.

## [0.3.0] - 2026-07-16

### Added
- Pretokenized sequence encoding (`encode-pretokenized`) preserving word IDs.
- Generic `builder` escape hatch to pass arbitrary DJL tokenizer options, and `from-bpe-files` construction from separate vocab.json + merges.txt.
- Opt-in `encode->ndlist` / `batch-encode->ndlist` for direct DJL inference pipelines, and NDManager-owned tokenizer lifecycle.
- Effective-config getters (`truncation`, `padding`, `max-length`, `stride`, `pad-to-multiple-of`, `effective-config`), `build-sentence`, and `native-version`.

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
