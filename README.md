# tokenizers-clj

Idiomatic Clojure tokenization: encode, decode, and count tokens against any
HuggingFace `tokenizer.json`, backed by the native Rust `tokenizers` library.

## Stack

<a href="https://clojure.org"><img src="https://img.shields.io/badge/Clojure-5881D8?style=flat&logo=clojure&logoColor=white" alt="Clojure" /></a>
<a href="https://www.java.com"><img src="https://img.shields.io/badge/JVM-ED8B00?style=flat&logo=openjdk&logoColor=white" alt="JVM" /></a>
<a href="https://huggingface.co/docs/tokenizers"><img src="https://img.shields.io/badge/HuggingFace%20Tokenizers-FFD21E?style=flat&logo=huggingface&logoColor=000" alt="HuggingFace Tokenizers" /></a>

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.savya/tokenizers-clj.svg)](https://clojars.org/net.clojars.savya/tokenizers-clj)
[![cljdoc](https://cljdoc.org/badge/net.clojars.savya/tokenizers-clj)](https://cljdoc.org/d/net.clojars.savya/tokenizers-clj)

A thin Clojure wrapper over [DJL](https://djl.ai/)'s
`ai.djl.huggingface/tokenizers`, which binds the same fast Rust
[`tokenizers`](https://github.com/huggingface/tokenizers) that HuggingFace ships
for Python. It gives you exact token counts and ids for BERT, GPT, Llama, Qwen,
and any other model that publishes a `tokenizer.json` - no Python, no network at
runtime once the model file is local.

## Install

Leiningen / Boot:

```clojure
[net.clojars.savya/tokenizers-clj "0.1.0"]
```

deps.edn:

```clojure
net.clojars.savya/tokenizers-clj {:mvn/version "0.1.0"}
```

## Usage

```clojure
(require '[tokenizers.core :as tok])

;; From a local tokenizer.json ...
(with-open [t (tok/from-file "bert-base-uncased/tokenizer.json")]
  (tok/count-tokens t "Hello, world!"))          ;=> 6

;; ... or straight from the HuggingFace hub (downloads + caches once).
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/encode t "Hello, world!"))
;=> {:ids [101 7592 1010 2088 999 102]
;    :tokens ["[CLS]" "hello" "," "world" "!" "[SEP]"]
;    :attention-mask [1 1 1 1 1 1]
;    :type-ids [0 0 0 0 0 0] :word-ids [...] :special-tokens-mask [1 0 0 0 0 1]}

;; Drop the framing special tokens for a raw count:
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/count-tokens t "Hello, world!" {:add-special-tokens? false}))  ;=> 4

;; Round-trip:
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/decode t (tok/ids t "hello there" {:add-special-tokens? false})))  ;=> "hello there"
```

`batch-encode` pads every sequence to the batch's longest so the result is
rectangular; real token counts are recoverable from each `:attention-mask`.

## Requirements

- JDK 8+.
- **A JVM matching your CPU architecture.** DJL loads a native library for the
  JVM's reported `os.arch`, so on Apple Silicon use an **arm64** JDK - an x86_64
  JVM running under Rosetta fails to resolve the native tokenizer
  (`Unexpected flavor: cpu`).
- Network access the first time DJL fetches the native library (cached
  afterwards under `~/.djl.ai/`), and on `from-pretrained` to download the model
  file.

## License

Copyright © 2026 Savyasachi

Distributed under the [Eclipse Public License 2.0](https://www.eclipse.org/legal/epl-2.0/).

Wraps [Deep Java Library](https://djl.ai/) (Apache-2.0) and the HuggingFace
[`tokenizers`](https://github.com/huggingface/tokenizers) library (Apache-2.0).
