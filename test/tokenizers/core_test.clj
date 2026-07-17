(ns tokenizers.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.io :as io]
            [tokenizers.core :as tok])
  (:import [ai.djl.ndarray NDList NDManager]
           [ai.djl.ndarray.types DataType]
           [com.sun.net.httpserver HttpHandler HttpServer]
           [java.net InetSocketAddress URI]
           [java.nio.file Files OpenOption StandardCopyOption]
           [java.nio.file.attribute FileAttribute]))

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
      (is (true? (:exceed-max-length? (tok/encode t "one two three"))))))
  (testing "stream construction accepts the same options"
    (with-open [stream (io/input-stream fixture)
                t (tok/from-stream stream {:add-special-tokens? false})]
      (is (= [7592] (tok/ids t "hello"))))))

(deftest effective-native-tokenizer-config
  (let [truncation (resolve 'tokenizers.core/truncation)
        padding (resolve 'tokenizers.core/padding)
        max-length (resolve 'tokenizers.core/max-length)
        stride (resolve 'tokenizers.core/stride)
        pad-to-multiple-of (resolve 'tokenizers.core/pad-to-multiple-of)
        effective-config (resolve 'tokenizers.core/effective-config)]
    (is (every? some? [truncation padding max-length stride
                       pad-to-multiple-of effective-config]))
    (when (every? some? [truncation padding max-length stride
                         pad-to-multiple-of effective-config])
      (with-open [t (tok/from-file fixture {:truncation :only-first
                                            :max-length 12
                                            :stride 3
                                            :padding :max-length
                                            :pad-to-multiple-of 4})]
        (is (= :only-first (truncation t)))
        (is (= :max-length (padding t)))
        (is (= 12 (max-length t)))
        (is (= 3 (stride t)))
        (is (= 4 (pad-to-multiple-of t)))
        (is (= {:truncation :only-first
                :padding :max-length
                :max-length 12
                :stride 3
                :pad-to-multiple-of 4}
               (effective-config t)))))))

(deftest raw-builder-options-pass-through
  (with-open [t (tok/from-file fixture
                               {:options {:modelMaxLength 2
                                          :addSpecialTokens false}})]
    (let [enc (tok/encode t "one two three")]
      (is (= [2028 2048] (:ids enc)))
      (is (true? (:exceed-max-length? enc))))))

(deftest tokenizer-can-be-owned-by-an-ndmanager
  (let [manager (NDManager/newBaseManager)
        t (tok/from-file fixture {:manager manager})]
    (try
      (is (false? (.isReleased t)))
      (finally
        (.close manager)))
    (is (true? (.isReleased t)))))

(deftest direct-bpe-vocab-and-merges-construction
  (let [from-bpe-files (resolve 'tokenizers.core/from-bpe-files)
        dir (Files/createTempDirectory "tokenizers-clj-bpe"
                                       (make-array FileAttribute 0))
        vocab (.resolve dir "vocab.json")
        merges (.resolve dir "merges.txt")]
    (Files/writeString
     vocab
     "{\"<unk>\":0,\"h\":1,\"e\":2,\"l\":3,\"o\":4,\"he\":5,\"hel\":6,\"hell\":7,\"hello\":8}"
     (make-array OpenOption 0))
    (Files/writeString merges "#version: 0.2\nh e\nhe l\nhel l\nhell o\n"
                       (make-array OpenOption 0))
    (is from-bpe-files)
    (when from-bpe-files
      (with-open [t (from-bpe-files vocab merges)]
        (let [enc (tok/encode t "hello" {:add-special-tokens? false})]
          (is (= [8] (:ids enc)))
          (is (= ["hello"] (:tokens enc))))))))

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

(deftest pretokenized-sequence-encode-preserves-word-ids
  (let [encode-pretokenized (resolve 'tokenizers.core/encode-pretokenized)]
    (is encode-pretokenized)
    (when encode-pretokenized
      (with-open [t (tok/from-file fixture)]
        (let [enc (encode-pretokenized t ["hello" "worlds"])]
          (is (= ["[CLS]" "hello" "worlds" "[SEP]"] (:tokens enc)))
          (is (= [-1 0 1 -1] (:word-ids enc)))))
      (with-open [t (tok/from-file fixture)]
        (is (= [7592 8484]
               (:ids (encode-pretokenized
                      t ["hello" "worlds"] {:add-special-tokens? false}))))))))

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

(deftest encoding-to-ndlist-for-inference
  (let [encode->ndlist (resolve 'tokenizers.core/encode->ndlist)]
    (is encode->ndlist)
    (when encode->ndlist
      (with-open [manager (NDManager/newBaseManager)
                  t (tok/from-file fixture)
                  nd-list (encode->ndlist
                           t "hello" manager
                           {:with-token-type-ids? true :int32? true})]
        (is (instance? NDList nd-list))
        (is (= 3 (count nd-list)))
        (is (= [101 7592 102] (vec (.toIntArray (.head nd-list)))))
        (is (= DataType/INT32 (.getDataType (.head nd-list))))))))

(deftest batch-encoding-to-ndlist-for-inference
  (let [batch-encode->ndlist (resolve 'tokenizers.core/batch-encode->ndlist)]
    (is batch-encode->ndlist)
    (when batch-encode->ndlist
      (with-open [manager (NDManager/newBaseManager)
                  t (tok/from-file fixture)
                  nd-list (batch-encode->ndlist
                           t ["hello" "world"] manager
                           {:with-token-type-ids? true})]
        (is (instance? NDList nd-list))
        (is (= 3 (count nd-list)))
        (is (= [101 7592 102 101 2088 102]
               (vec (.toLongArray (.head nd-list)))))))))

(deftest batch-decode-many-id-sequences
  (let [batch-decode (resolve 'tokenizers.core/batch-decode)]
    (is batch-decode)
    (when batch-decode
      (with-open [t (tok/from-file fixture)]
        (let [id-seqs (mapv :ids (tok/batch-encode t ["hello" "world"]))]
          (is (= ["hello" "world"] (batch-decode t id-seqs)))
          (is (every? #(re-find #"\[CLS\]" %)
                      (batch-decode t id-seqs {:skip-special-tokens? false}))))))))

(deftest from-pretrained-uses-revisioned-local-cache-offline
  (let [cache (Files/createTempDirectory "tokenizers-clj-cache"
                                         (make-array FileAttribute 0))
        cached-dir (.resolve cache "acme%2Fmodel/abc123")
        cached-tokenizer (.resolve cached-dir "tokenizer.json")]
    (Files/createDirectories cached-dir (make-array FileAttribute 0))
    (Files/copy (.toPath fixture) cached-tokenizer
                (into-array java.nio.file.CopyOption
                            [StandardCopyOption/REPLACE_EXISTING]))
    (with-open [t (tok/from-pretrained "acme/model"
                                       {:revision "abc123"
                                        :cache-dir cache
                                        :local-only? true
                                        :add-special-tokens? false})]
      (is (= [7592] (tok/ids t "hello"))))
    (with-open [t (tok/from-pretrained "acme/model"
                                       {:revision "abc123"
                                        :cache-dir cache
                                        :offline? true})]
      (is (= [101 7592 102] (tok/ids t "hello"))))))

(deftest from-pretrained-offline-requires-cached-revision
  (let [cache (Files/createTempDirectory "tokenizers-clj-empty-cache"
                                         (make-array FileAttribute 0))]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"not found in the local cache"
         (tok/from-pretrained "acme/missing"
                              {:revision "deadbeef"
                               :cache-dir cache
                               :local-only? true})))))

(deftest from-pretrained-builds-pinned-hub-uri
  (let [hub-uri (resolve 'tokenizers.core/hub-uri)]
    (is hub-uri)
    (when hub-uri
      (is (= "https://huggingface.co/acme/model/resolve/refs%2Fpr%2F7/tokenizer.json"
             (str (hub-uri "acme/model" "refs/pr/7")))))))

(deftest hub-download-sends-auth-and-populates-cache
  (let [download-tokenizer! (resolve 'tokenizers.core/download-tokenizer!)
        received-auth (atom nil)
        body (Files/readAllBytes (.toPath fixture))
        server (HttpServer/create (InetSocketAddress. 0) 0)
        target-dir (Files/createTempDirectory "tokenizers-clj-download"
                                              (make-array FileAttribute 0))
        target (.resolve target-dir "tokenizer.json")]
    (is download-tokenizer!)
    (.createContext
     server "/tokenizer.json"
     (reify HttpHandler
       (handle [_ exchange]
         (reset! received-auth (.getFirst (.getRequestHeaders exchange) "Authorization"))
         (.sendResponseHeaders exchange 200 (alength body))
         (with-open [out (.getResponseBody exchange)]
           (.write out body)))))
    (.start server)
    (try
      (when download-tokenizer!
        (download-tokenizer!
         (URI/create (str "http://127.0.0.1:" (.getPort (.getAddress server))
                          "/tokenizer.json"))
         target "hf_secret")
        (is (= "Bearer hf_secret" @received-auth))
        (is (= (seq body) (seq (Files/readAllBytes target)))))
      (finally
        (.stop server 0)))))

(deftest native-runtime-preflight-explains-macos-x86-jvm
  (let [check (resolve 'tokenizers.core/assert-compatible-native-runtime!)]
    (is check)
    (when check
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"requires an arm64/aarch64 JVM on macOS"
           (check "Mac OS X" "x86_64"))))))
