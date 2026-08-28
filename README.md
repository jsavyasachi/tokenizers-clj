# tokenizers-clj

[![Clojars Project](https://img.shields.io/clojars/v/net.clojars.savya/tokenizers-clj.svg)](https://clojars.org/net.clojars.savya/tokenizers-clj)
[![cljdoc](https://cljdoc.org/badge/net.clojars.savya/tokenizers-clj)](https://cljdoc.org/d/net.clojars.savya/tokenizers-clj)
[![test](https://github.com/jsavyasachi/tokenizers-clj/actions/workflows/test.yml/badge.svg)](https://github.com/jsavyasachi/tokenizers-clj/actions/workflows/test.yml)

Idiomatic Clojure tokenization: tokenize, encode, decode, and count tokens against any
HuggingFace `tokenizer.json`, backed by the native Rust `tokenizers` library.

## Stack

<a href="https://clojure.org"><img src="https://img.shields.io/badge/Clojure-5881D8?style=flat&logo=clojure&logoColor=white" alt="Clojure" /></a>
<a href="https://clojure.org/guides/deps_and_cli"><img src="https://img.shields.io/badge/deps.edn-5881D8?style=flat&logo=clojure&logoColor=fff" alt="deps.edn" /></a>
<a href="https://clojure.github.io/tools.build/"><img src="https://img.shields.io/badge/tools.build-5881D8?style=flat&logo=clojure&logoColor=fff" alt="tools.build" /></a>
<a href="https://www.java.com"><img src="https://img.shields.io/badge/JVM-ED8B00?style=flat&logo=openjdk&logoColor=white" alt="JVM" /></a>
<a href="https://huggingface.co/docs/tokenizers"><img src="https://img.shields.io/badge/HuggingFace%20Tokenizers-FFD21E?style=flat&logo=huggingface&logoColor=000" alt="HuggingFace Tokenizers" /></a>

A thin Clojure wrapper over [DJL](https://djl.ai/)'s
`ai.djl.huggingface/tokenizers`. DJL binds the same Rust
[`tokenizers`](https://github.com/huggingface/tokenizers) library that
HuggingFace ships for Python. The wrapper gives you exact token counts and ids
for BERT, GPT, Llama, Qwen, and any other model that publishes a
`tokenizer.json`.

## Install

deps.edn:

```clojure
net.clojars.savya/tokenizers-clj {:mvn/version "0.8.0"}
```

Leiningen / Boot:

```clojure
[net.clojars.savya/tokenizers-clj "0.8.0"]
```

## Usage

```clojure
(require '[tokenizers.core :as tok])

;; From a local tokenizer.json ...
(with-open [t (tok/from-file "bert-base-uncased/tokenizer.json")]
  (tok/count-tokens t "Hello, world!"))          ;=> 6

;; Get token strings directly, without an encode result map:
(with-open [t (tok/from-file "bert-base-uncased/tokenizer.json")]
  (tok/tokenize t "Hello, world!"))              ;=> ["[CLS]" "hello" "," "world" "!" "[SEP]"]

;; ... or straight from the HuggingFace hub (downloads + caches both tokenizer
;; files once, including tokenizer_config.json when the model provides it).
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
`:local-only?` / `:offline?`. If you supply a revision, cache, or offline option,
the library uses a revision-specific local cache. In an offline mode the library
fails without a network request when the tokenizer is absent.

Hub loading also downloads `tokenizer_config.json` when available and passes it
to DJL. Its `model_max_length` is reflected by `max-length`,
`effective-config`, and `:exceed-max-length?`; BOS/EOS/UNK/PAD and other special
token metadata is applied by DJL during tokenization and padding. A missing
`tokenizer_config.json` remains compatible with repositories that publish only
`tokenizer.json`.

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

Count native batch results without encode maps:

```clojure
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/batch-count-tokens t ["hello" "world"]))
;=> [3 3]
```

Span helpers operate directly on an `encode` result:

```clojure
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (let [enc (tok/encode t "unaffordable cat")]
    [(tok/token->chars enc 2)
     (tok/token->word enc 2)
     (tok/char->token enc 3)
     (tok/word->tokens enc 0)]))
;=> [[3 5] 0 2 [1 2 3 4]]
```

### Token budgets and text windows

`split-by-token-budget` and `truncate-by-token-budget` measure text with the
real tokenizer, then return encode-shaped chunks with `:ids`, `:tokens`,
`:offsets`, `:text`, and `:offset`. Offsets are indexes into the original Java
string (UTF-16), including correct conversion around supplementary characters.
The budget includes special tokens by default; set `:count-special-tokens?`
to false to exclude tokenizer-added special tokens while retaining them in the
first/last chunk. Chunks expose `:overflow?`, `:overflow-token-count`, and
`:overflow-token-ids`. `split-by-tokens` and `truncate-by-tokens` are aliases.

```clojure
(with-open [t (tok/from-pretrained "bert-base-uncased")]
  (tok/split-by-token-budget t "A long passage" 128
                              {:count-special-tokens? true}))
```

Batch encode options are the same as `encode` options. `batch-decode` accepts
`:skip-special-tokens?`, which defaults to true. Padding set at construction can
make batch results rectangular. You can get the real token counts from each
`:attention-mask`.

### Production patterns

Decoder-only tokenizers usually do not add BERT-style framing by default. Check
the model's tokenizer configuration and make the choice explicit:

```clojure
;; GPT-2: raw text commonly needs no BOS/EOS framing.
(with-open [t (tok/from-pretrained "gpt2")]
  (tok/ids t "Complete this" {:add-special-tokens? false}))

;; Llama and Qwen deployments often require BOS/EOS according to the model
;; chat template. Use the published tokenizer.json/config as the source of truth.
(with-open [t (tok/from-pretrained "meta-llama/Llama-3.2-1B")]
  (tok/encode t "Answer briefly" {:add-special-tokens? true}))
(with-open [t (tok/from-pretrained "Qwen/Qwen2.5-0.5B")]
  (tok/encode t "Answer briefly" {:add-special-tokens? true}))
```

For offline deployment, download both tokenizer files during image building and
point `:cache-dir` at the copied cache. Pin the revision and require offline
loading at runtime so startup cannot silently reach the network:

```clojure
(def tokenizer
  (tok/from-pretrained
   "bert-base-uncased"
   {:revision "<commit-sha>"
    :cache-dir "/opt/models/tokenizers"
    :offline? true
    :padding :max-length
    :max-length 128}))
```

For DJL inference, keep the tokenizer and manager lifecycle together. Batch
padding makes each row rectangular; padded positions have attention-mask `0`,
so the model should receive the mask rather than treating padding as content:

```clojure
(with-open [manager (NDManager/newBaseManager)
            t (tok/from-pretrained "bert-base-uncased")
            inputs (tok/batch-encode->ndlist
                    t ["short input" "a longer input sequence"] manager)]
  ;; inputs contains input ids and attention mask (plus token type ids when
  ;; :with-token-type-ids? is true); pass it directly to Predictor/Block.
  (.forward block inputs))
```

## Requirements

- JDK 17 or newer
- **A JVM that matches your CPU architecture.** DJL loads a native library for
  the JVM's reported `os.arch`. On Apple Silicon, use an **arm64** JDK. An
  x86_64 JVM under Rosetta cannot resolve the native tokenizer and fails with
  `Unexpected flavor: cpu`. Check the JVM with
  `java -XshowSettings:properties -version 2>&1 | grep 'os.arch\|java.home'`.
- Network access the first time DJL gets the native library. DJL then caches it
  under `~/.djl.ai/`. `from-pretrained` also needs network access to download
  the model file.

## License

Copyright © 2026 Savyasachi

Distributed under the [Eclipse Public License 2.0](https://www.eclipse.org/legal/epl-2.0/).

Wraps [Deep Java Library](https://djl.ai/) (Apache-2.0) and the HuggingFace
[`tokenizers`](https://github.com/huggingface/tokenizers) library (Apache-2.0).
