(ns tokenizers.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.io :as io]
            [tokenizers.core :as tok]))

(def fixture
  "bert-base-uncased tokenizer.json, vendored so the suite is offline + deterministic.
  (DJL still fetches the native JNI lib once on first encode, then caches it.)"
  (io/file (io/resource "bert-base-uncased-tokenizer.json")))

(def config-fixture
  (io/file (io/resource "tokenizer_config.json")))

;; bert-base-uncased lowercases and frames with [CLS]/[SEP]; these ids are stable.
(deftest encode-shape
  (with-open [t (tok/from-file fixture)]
    (let [enc (tok/encode t "Hello, world!")]
      (is (= [101 7592 1010 2088 999 102] (:ids enc)))
      (is (= ["[CLS]" "hello" "," "world" "!" "[SEP]"] (:tokens enc)))
      (is (= [1 1 1 1 1 1] (:attention-mask enc)))
      (is (= 6 (count (:type-ids enc)))))))

(deftest encode-exposes-source-attribution-and-overflow-metadata
  (with-open [t (tok/from-file fixture)]
    (let [enc (tok/encode t "Hello, world!")]
      (is (= [nil [0 5] [5 6] [7 12] [12 13] nil]
             (:offsets enc)))
      (is (= [-1 0 0 0 0 -1] (:sequence-ids enc)))
      (is (= [] (:overflow enc)))
      (is (false? (:exceed-max-length? enc))))))

(deftest count-tokens-and-special-tokens-toggle
  (with-open [t (tok/from-file fixture)]
    (is (= 6 (tok/count-tokens t "Hello, world!")))
    (testing "dropping [CLS]/[SEP] removes two tokens"
      (is (= 4 (tok/count-tokens t "Hello, world!" {:add-special-tokens? false})))
      (is (= [7592 1010 2088 999]
             (tok/ids t "Hello, world!" {:add-special-tokens? false}))))))

(deftest configurable-tokenizer-construction
  (testing "truncation, stride, overflow, and padding"
    (with-open [t (tok/from-file fixture {:truncation :longest-first
                                          :max-length 8
                                          :stride 2
                                          :padding :max-length
                                          :pad-to-multiple-of 4
                                          :with-overflowing-tokens? true})]
      (let [enc (tok/encode t "one two three four five six seven eight nine")]
        (is (= 8 (count (:ids enc))))
        (is (seq (:overflow enc)))
        (is (= "LONGEST_FIRST" (.getTruncation t)))
        (is (= "MAX_LENGTH" (.getPadding t)))
        (is (= 2 (.getStride t)))
        (is (= 4 (.getPadToMultipleOf t))))))
  (testing "constructor-level special-token and lowercase controls"
    (with-open [t (tok/from-file fixture {:add-special-tokens? false
                                          :lowercase? true})]
      (is (= [7592] (tok/ids t "HELLO")))))
  (testing "tokenizer_config.json supplies model max length"
    (with-open [t (tok/from-file fixture {:tokenizer-config config-fixture})]
      (is (true? (:exceed-max-length? (tok/encode t "one two three")))))))

(deftest decode-roundtrip
  (with-open [t (tok/from-file fixture)]
    (let [ids (tok/ids t "Hello, world!" {:add-special-tokens? false})
          text (tok/decode t ids)]
      (is (string? text))
      (is (re-find #"hello" text))
      (is (re-find #"world" text)))))

(deftest paired-sequence-encode
  (with-open [t (tok/from-file fixture)]
    (let [enc (tok/encode t "hello world" "goodbye friend")]
      (is (= ["[CLS]" "hello" "world" "[SEP]" "goodbye" "friend" "[SEP]"]
             (:tokens enc)))
      (is (= [0 0 0 0 1 1 1] (:type-ids enc)))
      (is (= [-1 0 0 -1 1 1 -1] (:sequence-ids enc))))
    (is (= [7592 2088 9119 2767]
           (:ids (tok/encode t "hello world" "goodbye friend"
                             {:add-special-tokens? false}))))))

(deftest paired-batch-encode
  (let [batch-encode-pairs (resolve 'tokenizers.core/batch-encode-pairs)]
    (is batch-encode-pairs)
    (when batch-encode-pairs
      (with-open [t (tok/from-file fixture)]
        (let [encs (batch-encode-pairs t [["hello" "world"]
                                           ["goodbye" "friend"]])]
          (is (= 2 (count encs)))
          (is (every? #(some #{1} (:type-ids %)) encs))
          (is (every? #(some #{1} (:sequence-ids %)) encs)))))))

(deftest batch-encode-pads-to-longest
  ;; DJL batchEncode pads every sequence to the batch's longest so the result is
  ;; rectangular: both :ids are length 5 here. Real token counts live in :attention-mask
  ;; (padding positions are 0). This is the key difference from (map encode texts).
  (with-open [t (tok/from-file fixture)]
    (let [encs (tok/batch-encode t ["hi" "hello there friend"])]
      (is (= 2 (count encs)))
      (is (every? #(= 5 (count (:ids %))) encs))
      (is (= 3 (reduce + (:attention-mask (first encs)))) "[CLS] hi [SEP]")
      (is (= 5 (reduce + (:attention-mask (second encs)))) "[CLS] hello there friend [SEP]"))))

(deftest batch-encode-accepts-encode-options
  (with-open [t (tok/from-file fixture)]
    (let [encs (tok/batch-encode t ["hello" "world"]
                                 {:add-special-tokens? false
                                  :with-overflowing-tokens? true})]
      (is (= [[7592] [2088]] (mapv :ids encs)))
      (is (= [[] []] (mapv :overflow encs))))))

(deftest batch-decode-many-id-sequences
  (let [batch-decode (resolve 'tokenizers.core/batch-decode)]
    (is batch-decode)
    (when batch-decode
      (with-open [t (tok/from-file fixture)]
        (let [id-seqs (mapv :ids (tok/batch-encode t ["hello" "world"]))]
          (is (= ["hello" "world"] (batch-decode t id-seqs)))
          (is (every? #(re-find #"\[CLS\]" %)
                      (batch-decode t id-seqs {:skip-special-tokens? false}))))))))

(deftest native-runtime-preflight-explains-macos-x86-jvm
  (let [check (resolve 'tokenizers.core/assert-compatible-native-runtime!)]
    (is check)
    (when check
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"requires an arm64/aarch64 JVM on macOS"
           (check "Mac OS X" "x86_64"))))))
