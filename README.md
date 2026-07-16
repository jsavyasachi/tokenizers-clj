# tokenizers-clj

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.savya/tokenizers-clj.svg)](https://clojars.org/net.clojars.savya/tokenizers-clj)
[![cljdoc](https://cljdoc.org/badge/net.clojars.savya/tokenizers-clj)](https://cljdoc.org/d/net.clojars.savya/tokenizers-clj)
[![test](https://github.com/jsavyasachi/tokenizers-clj/actions/workflows/test.yml/badge.svg)](https://github.com/jsavyasachi/tokenizers-clj/actions/workflows/test.yml)

Idiomatic Clojure tokenization: encode, decode, and count tokens against any
HuggingFace `tokenizer.json`, backed by the native Rust `tokenizers` library.

## Stack

<a href="https://clojure.org"><img src="https://img.shields.io/badge/Clojure-5881D8?style=flat&logo=clojure&logoColor=white" alt="Clojure" /></a>
<a href="https://clojure.org/guides/deps_and_cli"><img src="https://img.shields.io/badge/deps.edn-5881D8?style=flat&logo=clojure&logoColor=fff" alt="deps.edn" /></a>
<a href="https://clojure.github.io/tools.build/"><img src="https://img.shields.io/badge/tools.build-5881D8?style=flat&logo=clojure&logoColor=fff" alt="tools.build" /></a>
<a href="https://www.java.com"><img src="https://img.shields.io/badge/JVM-ED8B00?style=flat&logo=openjdk&logoColor=white" alt="JVM" /></a>
<a href="https://huggingface.co/docs/tokenizers"><img src="https://img.shields.io/badge/HuggingFace%20Tokenizers-FFD21E?style=flat&logo=huggingface&logoColor=000" alt="HuggingFace Tokenizers" /></a>

A thin Clojure wrapper over [DJL](https://djl.ai/)'s
`ai.djl.huggingface/tokenizers`, which binds the same fast Rust
[`tokenizers`](https://github.com/huggingface/tokenizers) that HuggingFace ships
for Python. It gives you exact token counts and ids for BERT, GPT, Llama, Qwen,
and any other model that publishes a `tokenizer.json`.

## Install

deps.edn:

```clojure
net.clojars.savya/tokenizers-clj {:mvn/version "0.2.0"}
```

Leiningen / Boot:

```clojure
[net.clojars.savya/tokenizers-clj "0.2.0"]
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
;    :type-ids [0 0 0 0 0 0] :word-ids [...]
;    :special-tokens-mask [1 0 0 0 0 1]
;    :offsets [[0 0] [0 5] [5 6] [7 12] [12 13] [0 0]]
;    :sequence-ids [...] :overflow [] :exceed-max-length? false}

;; Drop the framing special tokens for a raw count:
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/count-tokens t "Hello, world!" {:add-special-tokens? false}))  ;=> 4

;; Round-trip:
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/decode t (tok/ids t "hello there" {:add-special-tokens? false})))  ;=> "hello there"
```

### Construction options

`from-file`, `from-stream`, and `from-pretrained` accept an options map:

- `:truncation`: `:longest-first`, `:only-first`, `:only-second`, or `:none`
  (booleans are also accepted).
- `:max-length` and `:stride`: truncation size and overlap.
- `:padding`: `:longest`, `:max-length`, or `:none` (booleans are also accepted).
- `:pad-to-multiple-of`: pad encoded lengths to a multiple.
- `:add-special-tokens?`, `:with-overflowing-tokens?`, and `:lowercase?`: tokenizer
  behavior flags.
- `:tokenizer-config`: path, `File`, or `Path` to a `tokenizer_config.json`.

`from-pretrained` also accepts `:revision`, `:auth-token`, `:cache-dir`, and
`:local-only?` / `:offline?`. Supplying revision, cache, or offline options uses a
revision-specific local cache; offline modes fail without making a network request
when the tokenizer is absent.

```clojure
(with-open [t (tok/from-pretrained
               "bert-base-uncased"
               {:revision "main"
                :cache-dir ".cache/tokenizers"
                :truncation :longest-first
                :max-length 128
                :stride 16
                :padding :max-length
                :pad-to-multiple-of 8})]
  (tok/encode t "A question" "A paired answer"))
```

### Encode results and batches

Every `encode`, `batch-encode`, and `batch-encode-pairs` result contains `:ids`,
`:tokens`, `:type-ids`, `:word-ids`, `:attention-mask`, `:special-tokens-mask`,
`:offsets`, `:sequence-ids`, `:overflow`, and `:exceed-max-length?`. Overflow entries
have the same shape.

`encode` accepts a second string for paired-sequence encoding. Its four-argument form
also accepts `:add-special-tokens?` and `:with-overflowing-tokens?`.

```clojure
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/encode t "Question" "Answer"
              {:add-special-tokens? true
               :with-overflowing-tokens? false})
  (tok/batch-encode t ["first" "second"]
                    {:add-special-tokens? false})
  (tok/batch-encode-pairs t [["question 1" "answer 1"]
                             ["question 2" "answer 2"]])
  (tok/batch-decode t [[101 2034 102] [101 2117 102]]
                    {:skip-special-tokens? true}))
```

Batch encoding options are the same as `encode` options. `batch-decode` accepts
`:skip-special-tokens?`, which defaults to true. Padding configured at construction
can make batch results rectangular; real token counts are recoverable from each
`:attention-mask`.

## Requirements

- JDK 21
- **A JVM matching your CPU architecture.** DJL loads a native library for the
  JVM's reported `os.arch`, so on Apple Silicon use an **arm64** JDK - an x86_64
  JVM running under Rosetta fails to resolve the native tokenizer
  (`Unexpected flavor: cpu`); check with
  `java -XshowSettings:properties -version 2>&1 | grep 'os.arch\|java.home'`.
- Network access the first time DJL fetches the native library (cached
  afterwards under `~/.djl.ai/`), and on `from-pretrained` to download the model
  file

## License

Copyright © 2026 Savyasachi

Distributed under the [Eclipse Public License 2.0](https://www.eclipse.org/legal/epl-2.0/).

Wraps [Deep Java Library](https://djl.ai/) (Apache-2.0) and the HuggingFace
[`tokenizers`](https://github.com/huggingface/tokenizers) library (Apache-2.0).
