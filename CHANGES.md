# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-06-29

### Added
- Preflight check for macOS x86_64 JVMs before DJL loads the native tokenizer,
  with an actionable Apple Silicon arm64 JDK message.

## [0.1.0] - 2026-06-22

Initial release.

### Added
- `from-file`, `from-pretrained`, `from-stream` constructors over DJL's
  `HuggingFaceTokenizer` (native Rust `tokenizers` via JNI).
- `encode` returning a Clojure map (`:ids :tokens :type-ids :word-ids
  :attention-mask :special-tokens-mask`), with `:add-special-tokens?` /
  `:with-overflowing-tokens?` options.
- `ids`, `tokens`, `count-tokens`, `decode`, and `batch-encode` helpers.
- Tokenizers are `Closeable`, so `with-open` frees the native handle.

[0.1.2]: https://github.com/jsavyasachi/tokenizers-clj/releases/tag/v0.1.2
[0.1.0]: https://github.com/jsavyasachi/tokenizers-clj/releases/tag/0.1.0
