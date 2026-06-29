(ns tokenizers.core
  "Idiomatic Clojure wrapper over DJL's HuggingFace tokenizers, which bind the native
  Rust `tokenizers` library via JNI. Build a tokenizer with `from-file` /
  `from-pretrained` / `from-stream`, then `encode`, `decode`, or `count-tokens`.

  A tokenizer holds a native handle: close it (`with-open` works) to free it."
  (:import [ai.djl.huggingface.tokenizers HuggingFaceTokenizer Encoding]
           [java.io File InputStream]
           [java.nio.file Path]
           [java.util HashMap Locale]))

(defn- ^Path as-path [x]
  (cond
    (instance? Path x) x
    (instance? File x) (.toPath ^File x)
    :else (.toPath (File. (str x)))))

(defn- lower-prop [^String s]
  (some-> s (.toLowerCase Locale/ROOT)))

(defn assert-compatible-native-runtime!
  "Fail early for known DJL native tokenizer runtime mismatches."
  ([]
   (assert-compatible-native-runtime! (System/getProperty "os.name")
                                      (System/getProperty "os.arch")))
  ([os-name os-arch]
   (let [os-name (lower-prop os-name)
         os-arch (lower-prop os-arch)]
     (when (and (some-> ^String os-name (.contains "mac"))
                (= "x86_64" os-arch))
       (throw
        (ex-info
         (str "tokenizers-clj requires an arm64/aarch64 JVM on macOS. "
              "DJL 0.36.0 does not ship an osx-x86_64 native tokenizer library, "
              "so an x86_64 JVM fails later with `Unexpected flavor: cpu`. "
              "Install and select an arm64 JDK, then retry.")
         {:os-name os-name
          :os-arch os-arch
          :expected-os-arch "aarch64"}))))))

(defn from-file
  "Tokenizer from a `tokenizer.json` (path string, `File`, or `Path`)."
  ^HuggingFaceTokenizer [path]
  (assert-compatible-native-runtime!)
  (HuggingFaceTokenizer/newInstance (as-path path)))

(defn from-pretrained
  "Tokenizer by HuggingFace hub id, e.g. \"bert-base-uncased\". Downloads then caches.
  Needs network on first use."
  ^HuggingFaceTokenizer [^String id]
  (assert-compatible-native-runtime!)
  (HuggingFaceTokenizer/newInstance id))

(defn from-stream
  "Tokenizer from an `InputStream` over a `tokenizer.json`."
  ^HuggingFaceTokenizer [^InputStream is]
  (assert-compatible-native-runtime!)
  (HuggingFaceTokenizer/newInstance is (HashMap.)))

(defn- enc->map [^Encoding e]
  {:ids (vec (.getIds e))
   :tokens (vec (.getTokens e))
   :type-ids (vec (.getTypeIds e))
   :word-ids (vec (.getWordIds e))
   :attention-mask (vec (.getAttentionMask e))
   :special-tokens-mask (vec (.getSpecialTokenMask e))})

(defn encode
  "Encode `text` into a map of `:ids :tokens :type-ids :word-ids :attention-mask
  :special-tokens-mask`. Opts: `:add-special-tokens?` (default true),
  `:with-overflowing-tokens?` (default false)."
  ([^HuggingFaceTokenizer t ^String text]
   (enc->map (.encode t text)))
  ([^HuggingFaceTokenizer t ^String text {:keys [add-special-tokens? with-overflowing-tokens?]
                                          :or {add-special-tokens? true
                                               with-overflowing-tokens? false}}]
   (enc->map (.encode t text (boolean add-special-tokens?) (boolean with-overflowing-tokens?)))))

(defn ids
  "Token ids for `text` (see `encode` for opts)."
  ([t text] (:ids (encode t text)))
  ([t text opts] (:ids (encode t text opts))))

(defn tokens
  "Token strings for `text` (see `encode` for opts)."
  ([t text] (:tokens (encode t text)))
  ([t text opts] (:tokens (encode t text opts))))

(defn count-tokens
  "Number of token ids `text` encodes to (see `encode` for opts)."
  ([t text] (count (ids t text)))
  ([t text opts] (count (ids t text opts))))

(defn decode
  "Decode a seq of token `id-seq` back to text. Opts: `:skip-special-tokens?` (default true)."
  ([^HuggingFaceTokenizer t id-seq]
   (.decode t (long-array id-seq)))
  ([^HuggingFaceTokenizer t id-seq {:keys [skip-special-tokens?] :or {skip-special-tokens? true}}]
   (.decode t (long-array id-seq) (boolean skip-special-tokens?))))

(defn batch-encode
  "Encode many `texts` at once, returning a vector of `encode`-shaped maps."
  [^HuggingFaceTokenizer t texts]
  (mapv enc->map (.batchEncode t ^java.util.List (vec texts))))
