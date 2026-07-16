(ns tokenizers.core
  "Idiomatic Clojure wrapper over DJL's HuggingFace tokenizers, which bind the native
  Rust `tokenizers` library via JNI. Build a tokenizer with `from-file` /
  `from-pretrained` / `from-stream`, then `encode`, `decode`, or `count-tokens`.

  A tokenizer holds a native handle: close it (`with-open` works) to free it."
  (:import [ai.djl.huggingface.tokenizers HuggingFaceTokenizer Encoding TokenizerConfig]
           [ai.djl.huggingface.tokenizers.jni CharSpan]
           [java.io File InputStream]
           [java.nio.file Path]
           [java.util Locale]))

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

(defn- option-value [value choices option]
  (or (get choices value)
      (throw (ex-info (str "Unsupported " option ": " (pr-str value))
                      {:option option :value value :supported (vec (keys choices))}))))

(defn- djl-options [opts]
  (let [truncation (if (contains? opts :truncation)
                     (:truncation opts)
                     (:truncation? opts))
        padding (if (contains? opts :padding)
                  (:padding opts)
                  (:padding? opts))]
    (update-vals
     (cond-> {}
       (some? truncation)
       (assoc "truncation" (option-value truncation
                                          {true "LONGEST_FIRST"
                                           false "DO_NOT_TRUNCATE"
                                           :longest-first "LONGEST_FIRST"
                                           :only-first "ONLY_FIRST"
                                           :only-second "ONLY_SECOND"
                                           :none "DO_NOT_TRUNCATE"}
                                          :truncation))

       (some? padding)
       (assoc "padding" (option-value padding
                                       {true "LONGEST"
                                        false "DO_NOT_PAD"
                                        :longest "LONGEST"
                                        :max-length "MAX_LENGTH"
                                        :none "DO_NOT_PAD"}
                                       :padding))

       (contains? opts :max-length)
       (assoc "maxLength" (:max-length opts))

       (contains? opts :stride)
       (assoc "stride" (:stride opts))

       (contains? opts :pad-to-multiple-of)
       (assoc "padToMultipleOf" (:pad-to-multiple-of opts))

       (contains? opts :add-special-tokens?)
       (assoc "addSpecialTokens" (:add-special-tokens? opts))

       (contains? opts :with-overflowing-tokens?)
       (assoc "withOverflowingTokens" (:with-overflowing-tokens? opts))

       (contains? opts :lowercase?)
       (assoc "doLowerCase" (:lowercase? opts)))
     str)))

(defn- tokenizer-config [opts]
  (some-> (:tokenizer-config opts) as-path TokenizerConfig/load))

(defn from-file
  "Tokenizer from a `tokenizer.json` (path string, `File`, or `Path`).
  Constructor options include `:truncation`, `:max-length`, `:stride`, `:padding`,
  `:pad-to-multiple-of`, `:add-special-tokens?`, `:lowercase?`, and
  `:tokenizer-config`."
  (^HuggingFaceTokenizer [path]
   (from-file path {}))
  (^HuggingFaceTokenizer [path opts]
   (assert-compatible-native-runtime!)
   (let [path (as-path path)
         options (djl-options opts)]
     (if-let [config (:tokenizer-config opts)]
       (HuggingFaceTokenizer/newInstance path (str (as-path config)) options)
       (HuggingFaceTokenizer/newInstance path options)))))

(defn from-pretrained
  "Tokenizer by HuggingFace hub id, e.g. \"bert-base-uncased\". Downloads then caches.
  Needs network on first use."
  (^HuggingFaceTokenizer [^String id]
   (from-pretrained id {}))
  (^HuggingFaceTokenizer [^String id opts]
   (assert-compatible-native-runtime!)
   (HuggingFaceTokenizer/newInstance id (djl-options opts))))

(defn from-stream
  "Tokenizer from an `InputStream` over a `tokenizer.json`, with constructor opts."
  (^HuggingFaceTokenizer [^InputStream is]
   (from-stream is {}))
  (^HuggingFaceTokenizer [^InputStream is opts]
   (assert-compatible-native-runtime!)
   (let [options (djl-options opts)]
     (if-let [config (tokenizer-config opts)]
       (HuggingFaceTokenizer/newInstance is options config)
       (HuggingFaceTokenizer/newInstance is options)))))

(defn- span->offset [^CharSpan span]
  (when span
    [(.getStart span) (.getEnd span)]))

(defn- enc->map [^Encoding e]
  {:ids (vec (.getIds e))
   :tokens (vec (.getTokens e))
   :type-ids (vec (.getTypeIds e))
   :word-ids (vec (.getWordIds e))
   :attention-mask (vec (.getAttentionMask e))
   :special-tokens-mask (vec (.getSpecialTokenMask e))
   :offsets (mapv span->offset (.getCharTokenSpans e))
   :sequence-ids (vec (.getSequenceIds e))
   :overflow (mapv enc->map (.getOverflowing e))
   :exceed-max-length? (.exceedMaxLength e)})

(defn encode
  "Encode `text` into a map of token data, offsets, sequence ids, overflow
  encodings, and max-length status. Opts: `:add-special-tokens?` (default true),
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
