(ns tokenizers.core
  "Idiomatic Clojure wrapper over DJL's HuggingFace tokenizers, which bind the native
  Rust `tokenizers` library via JNI. Build a tokenizer with `from-file` /
  `from-pretrained` / `from-stream`, then `encode`, `decode`, or `count-tokens`.

  A tokenizer holds a native handle: close it (`with-open` works) to free it."
  (:require [clojure.string :as str])
  (:import [ai.djl.huggingface.tokenizers HuggingFaceTokenizer
            HuggingFaceTokenizer$Builder Encoding TokenizerConfig]
           [ai.djl.huggingface.tokenizers.jni CharSpan]
           [java.io File InputStream]
           [java.net URI URLEncoder]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest
            HttpResponse$BodyHandlers]
           [java.nio.charset StandardCharsets]
           [java.nio.file CopyOption Files LinkOption OpenOption Path StandardCopyOption]
           [java.nio.file.attribute FileAttribute]
           [java.time Duration]
           [java.util Locale]
           [ai.djl.util PairList]))

(defn- as-path ^Path [x]
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
    (into {}
          (map (fn [[key value]] [key (str value)]))
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
            (assoc "doLowerCase" (:lowercase? opts))))))

(defn- raw-options [opts]
  (into {}
        (map (fn [[key value]]
               [(if (keyword? key) (name key) (str key)) (str value)]))
        (merge (:options opts) (:raw-options opts))))

(defn- constructor-options ^java.util.Map [opts]
  (merge (djl-options opts) (raw-options opts)))

(defn builder
  "Create a DJL tokenizer builder configured from wrapper opts.
  Entries in `:options` or `:raw-options` pass through verbatim by DJL option
  name, such as `modelMaxLength`, `stripAccents`, and `addPrefixSpace`; keyword
  keys are converted with `name`. Raw entries override translated wrapper opts.
  `:manager` attaches the built tokenizer to a caller-supplied `NDManager`."
  (^HuggingFaceTokenizer$Builder []
   (HuggingFaceTokenizer/builder))
  (^HuggingFaceTokenizer$Builder [opts]
   (let [builder (HuggingFaceTokenizer/builder)]
     (.configure builder (constructor-options opts))
     (when-let [manager (:manager opts)]
       (.optManager builder manager))
     builder)))

(defn- tokenizer-config ^TokenizerConfig [opts]
  (some-> (:tokenizer-config opts) as-path TokenizerConfig/load))

(defn from-file
  "Tokenizer from a `tokenizer.json` (path string, `File`, or `Path`).
  Constructor options include `:truncation`, `:max-length`, `:stride`, `:padding`,
  `:pad-to-multiple-of`, `:add-special-tokens?`, `:lowercase?`, and
  `:tokenizer-config`. See `builder` for raw options and `:manager`."
  (^HuggingFaceTokenizer [path]
   (from-file path {}))
  (^HuggingFaceTokenizer [path opts]
   (assert-compatible-native-runtime!)
   (let [builder (builder opts)]
     (.optTokenizerPath builder (as-path path))
     (when-let [config (:tokenizer-config opts)]
       (.optTokenizerConfigPath builder (str (as-path config))))
     (.build builder))))

(defn- encode-component [value]
  (.replace (URLEncoder/encode (str value) StandardCharsets/UTF_8) "+" "%20"))

(defn- hub-uri
  ([id revision]
   (hub-uri id revision "tokenizer.json"))
  ([id revision filename]
  (URI/create
   (str "https://huggingface.co/"
        (str/replace (encode-component id) "%2F" "/")
        "/resolve/" (encode-component revision) "/" filename))))

(defn- hub-cache-path
  ([id revision cache-dir]
   (hub-cache-path id revision cache-dir "tokenizer.json"))
  ([id revision cache-dir filename]
  (-> (as-path (or cache-dir
                   (str (System/getProperty "user.home")
                        File/separator ".cache" File/separator
                        "huggingface" File/separator "tokenizers-clj")))
      (.resolve ^String (encode-component id))
      (.resolve ^String (encode-component revision))
      (.resolve filename))))

(def ^:private default-hub-download-timeout-ms 30000)

(defn- hub-download-timeout [timeout-ms]
  (let [timeout-ms (or timeout-ms default-hub-download-timeout-ms)]
    (when-not (and (integer? timeout-ms) (pos? timeout-ms))
      (throw (ex-info ":download-timeout-ms must be a positive integer"
                      {:download-timeout-ms timeout-ms})))
    (Duration/ofMillis timeout-ms)))

(defn- hub-http-client [^Duration timeout]
  (-> (HttpClient/newBuilder)
      (.connectTimeout timeout)
      (.followRedirects HttpClient$Redirect/ALWAYS)
      (.build)))

(defn- hub-request [uri auth-token ^Duration timeout]
  (let [request-builder (doto (HttpRequest/newBuilder uri)
                          (.GET)
                          (.timeout timeout))]
    (when auth-token
      (.header request-builder "Authorization" (str "Bearer " auth-token)))
    (.build request-builder)))

(defn- download-tokenizer!
  ([uri ^Path target auth-token]
   (download-tokenizer! uri target auth-token nil))
  ([uri ^Path target auth-token timeout-ms]
  (Files/createDirectories (.getParent target) (make-array FileAttribute 0))
  (let [timeout (hub-download-timeout timeout-ms)
        client (hub-http-client timeout)
        response (.send client (hub-request uri auth-token timeout)
                        (HttpResponse$BodyHandlers/ofByteArray))
        status (.statusCode response)]
    (when-not (<= 200 status 299)
      (throw (ex-info (str "HuggingFace Hub returned HTTP " status " for " uri)
                      {:status status :uri (str uri)})))
    (let [temp (Files/createTempFile (.getParent target) ".tokenizer-" ".json"
                                     (make-array FileAttribute 0))]
      (try
        (Files/write temp ^bytes (.body response)
                     ^"[Ljava.nio.file.OpenOption;" (make-array OpenOption 0))
        (Files/move temp target
                    (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))
        (finally
          (Files/deleteIfExists temp))))
    target)))

(def ^:private hub-option-keys
  #{:revision :auth-token :cache-dir :local-only? :local-only :offline? :offline
    :download-timeout-ms})

(defn- wrapper-managed-hub? [opts]
  (some #(contains? opts %) [:revision :cache-dir :local-only? :local-only
                             :offline? :offline :download-timeout-ms]))

(defn- offline? [opts]
  (boolean (or (:local-only? opts) (:local-only opts)
               (:offline? opts) (:offline opts))))

(defn from-pretrained
  "Tokenizer by HuggingFace Hub id. Options include `:revision`, `:auth-token`,
  `:cache-dir`, `:local-only?` / `:offline?`, and `:download-timeout-ms`.
  Wrapper-managed Hub downloads cache both `tokenizer.json` and the optional
  `tokenizer_config.json`; DJL applies the latter's model and special-token
  metadata during construction.
  Wrapper-managed Hub downloads use a 30,000 ms connect and read timeout by
  default; `:download-timeout-ms` must be a positive integer."
  (^HuggingFaceTokenizer [^String id]
   (from-pretrained id {}))
  (^HuggingFaceTokenizer [^String id opts]
   (assert-compatible-native-runtime!)
   (if (wrapper-managed-hub? opts)
     (let [revision (str (or (:revision opts) "main"))
           cache-dir (:cache-dir opts)
           path (hub-cache-path id revision cache-dir)
           config-path (hub-cache-path id revision cache-dir "tokenizer_config.json")]
       (when-not (Files/exists path (make-array LinkOption 0))
         (if (offline? opts)
           (throw (ex-info (str "Tokenizer " id " at revision " revision
                               " was not found in the local cache")
                           {:id id :revision revision :cache-path (str path)}))
           (download-tokenizer! (hub-uri id revision) path (:auth-token opts)
                                (:download-timeout-ms opts))))
       (when (and (not (Files/exists config-path (make-array LinkOption 0)))
                  (not (offline? opts)))
         (try
           (download-tokenizer! (hub-uri id revision "tokenizer_config.json")
                                config-path (:auth-token opts)
                                (:download-timeout-ms opts))
           (catch clojure.lang.ExceptionInfo error
             (when-not (= 404 (:status (ex-data error)))
               (throw error)))))
       (from-file path
                  (cond-> (apply dissoc opts hub-option-keys)
                    (Files/exists config-path (make-array LinkOption 0))
                    (assoc :tokenizer-config config-path))))
     (let [opts (cond-> opts
                  (:auth-token opts)
                  (update :raw-options merge
                          {"hf_token" (str (:auth-token opts))}))
           builder (builder opts)]
       (.optTokenizerName builder id)
       (.build builder)))))

(defn from-stream
  "Tokenizer from an `InputStream` over a `tokenizer.json`, with constructor opts."
  (^HuggingFaceTokenizer [^InputStream is]
   (from-stream is {}))
  (^HuggingFaceTokenizer [^InputStream is opts]
   (assert-compatible-native-runtime!)
   (let [^java.util.Map options (constructor-options opts)]
     (if-let [^TokenizerConfig config (tokenizer-config opts)]
       (HuggingFaceTokenizer/newInstance is options config)
       (HuggingFaceTokenizer/newInstance is options)))))

(defn from-bpe-files
  "BPE tokenizer from separate `vocab.json` and `merges.txt` paths.
  Accepts the constructor options documented by `from-file`, including raw
  `:options` / `:raw-options`."
  (^HuggingFaceTokenizer [vocab-path merges-path]
   (from-bpe-files vocab-path merges-path {}))
  (^HuggingFaceTokenizer [vocab-path merges-path opts]
   (assert-compatible-native-runtime!)
   (HuggingFaceTokenizer/newInstance (as-path vocab-path)
                                     (as-path merges-path)
                                     (constructor-options opts))))

(defn- strategy-keyword [value]
  (some-> value lower-prop (str/replace "_" "-") keyword))

(defn truncation
  "Effective native truncation strategy as a keyword."
  [^HuggingFaceTokenizer t]
  (strategy-keyword (.getTruncation t)))

(defn padding
  "Effective native padding strategy as a keyword."
  [^HuggingFaceTokenizer t]
  (strategy-keyword (.getPadding t)))

(defn max-length
  "Effective native maximum sequence length."
  [^HuggingFaceTokenizer t]
  (.getMaxLength t))

(defn stride
  "Effective native truncation stride."
  [^HuggingFaceTokenizer t]
  (.getStride t))

(defn pad-to-multiple-of
  "Effective native padding multiple."
  [^HuggingFaceTokenizer t]
  (.getPadToMultipleOf t))

(defn effective-config
  "Effective native truncation and padding configuration as a Clojure map."
  [^HuggingFaceTokenizer t]
  {:truncation (truncation t)
   :padding (padding t)
   :max-length (max-length t)
   :stride (stride t)
   :pad-to-multiple-of (pad-to-multiple-of t)})

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

(defn- raw-encode
  (^Encoding [^HuggingFaceTokenizer t ^String text]
   (.encode t text))
  (^Encoding [^HuggingFaceTokenizer t ^String text pair-or-opts]
   (if (map? pair-or-opts)
     (let [{:keys [add-special-tokens? with-overflowing-tokens?]
            :or {add-special-tokens? true with-overflowing-tokens? false}} pair-or-opts]
       (.encode t text (boolean add-special-tokens?)
                (boolean with-overflowing-tokens?)))
     (.encode t text ^String pair-or-opts)))
  (^Encoding [^HuggingFaceTokenizer t ^String text ^String text-pair
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (.encode t text text-pair (boolean add-special-tokens?)
            (boolean with-overflowing-tokens?))))

(defn encode
  "Encode `text`, optionally paired with a second text, into a token-data map.
  Opts: `:add-special-tokens?` (default true), `:with-overflowing-tokens?`
  (default false)."
  ([t text]
   (enc->map (raw-encode t text)))
  ([t text pair-or-opts]
   (enc->map (raw-encode t text pair-or-opts)))
  ([t text text-pair opts]
   (enc->map (raw-encode t text text-pair opts))))

(defn- validate-token-budget [token-budget]
  (when-not (and (integer? token-budget) (pos? token-budget))
    (throw (ex-info ":token-budget must be a positive integer"
                    {:token-budget token-budget})))
  token-budget)

(defn- token-records [enc]
  (mapv (fn [idx]
          {:idx idx
           :special? (= 1 (get (:special-tokens-mask enc) idx))})
        (range (count (:ids enc)))))

(defn- budget-groups [records token-budget count-special-tokens?]
  (loop [remaining records
         current []
         groups []]
    (if-let [record (first remaining)]
      (let [next-current (conj current record)
            weight (if (and (:special? record) (not count-special-tokens?)) 0 1)
            current-weight (reduce + (map #(if (and (:special? %) (not count-special-tokens?))
                                           0
                                           1)
                                          current))]
        (if (and (seq current) (> (+ current-weight weight) token-budget))
          (recur remaining [] (conj groups current))
          (recur (next remaining) next-current groups)))
      (cond-> groups
        (seq current) (conj current)))))

(defn- content-groups [records token-budget]
  (->> records
       (remove :special?)
       (partition-all token-budget)
       (mapv vec)))

(defn- attach-special-tokens [records groups]
  (let [prefix (take-while :special? records)
        suffix (take-while :special? (reverse records))]
    (if (seq groups)
      (cond-> (mapv vec groups)
        (seq prefix) (update 0 #(vec (concat prefix %)))
        (seq suffix) (update (dec (count groups)) #(vec (concat % (reverse suffix)))))
      (when (seq records)
        [(vec records)]))))

(defn- chunk-map [enc text records chunk-index chunk-count all-ids]
  (let [indices (mapv :idx records)
        offsets (mapv (fn [idx]
                        (when-let [[start end] (get (:offsets enc) idx)]
                          [(.offsetByCodePoints ^String text 0 (int start))
                           (.offsetByCodePoints ^String text 0 (int end))]))
                      indices)
        present-offsets (keep identity offsets)
        offset (when (seq present-offsets)
                 [(ffirst present-offsets) (second (last present-offsets))])
        overflow-token-ids (vec (drop (inc (apply max indices)) all-ids))]
    (assoc (reduce (fn [m key]
                     (assoc m key (mapv #(get-in enc [key %]) indices)))
                   (assoc enc :offsets offsets)
                   [:ids :tokens :type-ids :word-ids :attention-mask
                    :special-tokens-mask :sequence-ids])
           :text (if offset (subs text (first offset) (second offset)) "")
           :offset offset
           :char-offset offset
           :chunk-index chunk-index
           :chunk-count chunk-count
           :overflow? (< chunk-index (dec chunk-count))
           :overflow-token-count (count overflow-token-ids)
           :overflow-token-ids overflow-token-ids)))

(defn split-by-token-budget
  "Split `text` into native-token windows of at most `token-budget` tokens.

  Returns encode-shaped maps with `:text`, `:offset` (Java character indexes),
  `:ids`, and overflow metadata. Special tokens count against the budget by
  default. Set `:count-special-tokens?` to false to exclude them while keeping
  tokenizer-added prefix/suffix specials in the first/last window. Other opts
  are the `encode` options, notably `:add-special-tokens?`. DJL 0.36.0 has no
  per-call budget API, so the full native encoding is measured before windows
  are partitioned.
  "
  ([^HuggingFaceTokenizer t ^String text token-budget]
   (split-by-token-budget t text token-budget {}))
  ([^HuggingFaceTokenizer t ^String text token-budget opts]
   (validate-token-budget token-budget)
   (let [count-special-tokens? (get opts :count-special-tokens? true)
         enc (encode t text (dissoc opts :count-special-tokens?))
         records (token-records enc)
         groups (if count-special-tokens?
                  (budget-groups records token-budget count-special-tokens?)
                  (attach-special-tokens records
                                          (content-groups records token-budget)))]
     (if (seq groups)
       (mapv (fn [index group]
               (chunk-map enc text group index (count groups) (:ids enc)))
             (range)
             groups)
       []))))

(defn truncate-by-token-budget
  "Return the first `split-by-token-budget` window, including overflow metadata.
  Returns nil when `text` produces no tokens. Special-token accounting and
  encode options match `split-by-token-budget`."
  ([^HuggingFaceTokenizer t ^String text token-budget]
   (truncate-by-token-budget t text token-budget {}))
  ([^HuggingFaceTokenizer t ^String text token-budget opts]
   (first (split-by-token-budget t text token-budget opts))))

(def split-by-tokens split-by-token-budget)
(def truncate-by-tokens truncate-by-token-budget)

(defn token->chars
  "Character span for `token-idx` in an `encode` result map, or nil."
  [enc token-idx]
  (get (:offsets enc) token-idx))

(defn token->word
  "Word index for `token-idx` in an `encode` result map, or nil."
  [enc token-idx]
  (let [word-id (get (:word-ids enc) token-idx -1)]
    (when-not (= -1 word-id)
      word-id)))

(defn char->token
  "Token index containing `char-idx` in an `encode` result map, or nil."
  [enc char-idx]
  (first
   (keep-indexed
    (fn [token-idx span]
      (when (and span
                 (<= (first span) char-idx)
                 (< char-idx (second span)))
        token-idx))
    (:offsets enc))))

(defn word->tokens
  "Token indices for `word-id` in an `encode` result map."
  [enc word-id]
  (into []
        (keep-indexed (fn [token-idx token-word-id]
                        (when (and (not= -1 token-word-id)
                                   (= word-id token-word-id))
                          token-idx)))
        (:word-ids enc)))

(defn encode-pretokenized
  "Encode an already-split sequence of word strings. This keeps the native word ids.
  Opts: `:add-special-tokens?` (default true), `:with-overflowing-tokens?`
  (default false)."
  ([^HuggingFaceTokenizer t words]
   (enc->map (.encode t ^java.util.List (vec words))))
  ([^HuggingFaceTokenizer t words
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (enc->map (.encode t ^java.util.List (vec words)
                      (boolean add-special-tokens?)
                      (boolean with-overflowing-tokens?)))))

(defn encode->ndlist
  "Encode `text` directly to a DJL `NDList` owned by `manager`.
  Opts include the `encode` opts plus `:with-token-type-ids?` and `:int32?`
  (both default false). The caller owns and must close the `NDManager`."
  ([^HuggingFaceTokenizer t ^String text manager]
   (encode->ndlist t text manager {}))
  ([^HuggingFaceTokenizer t ^String text manager
    {:keys [add-special-tokens? with-overflowing-tokens?
            with-token-type-ids? int32?]
     :or {add-special-tokens? true
          with-overflowing-tokens? false
          with-token-type-ids? false
          int32? false}}]
   (-> (.encode t text (boolean add-special-tokens?)
                (boolean with-overflowing-tokens?))
       (.toNDList manager (boolean with-token-type-ids?) (boolean int32?)))))

(defn ids
  "Token ids for `text` (see `encode` for opts)."
  ([t text]
   (vec (.getIds ^Encoding (raw-encode t text))))
  ([t text opts]
   (vec (.getIds ^Encoding (raw-encode t text opts)))))

(defn tokens
  "Token strings for `text` (see `encode` for opts)."
  ([t text]
   (vec (.getTokens ^Encoding (raw-encode t text))))
  ([t text opts]
   (vec (.getTokens ^Encoding (raw-encode t text opts)))))

(defn tokenize
  "Token strings for `text` without constructing an encode result."
  [^HuggingFaceTokenizer t ^String text]
  (vec (.tokenize t text)))

(defn count-tokens
  "Number of token ids `text` encodes to (see `encode` for opts)."
  ([t text]
   (alength (.getIds ^Encoding (raw-encode t text))))
  ([t text opts]
   (alength (.getIds ^Encoding (raw-encode t text opts)))))

(defn decode
  "Decode a seq of token `id-seq` back to text. Opts: `:skip-special-tokens?` (default true)."
  ([^HuggingFaceTokenizer t id-seq]
   (.decode t (long-array id-seq)))
  ([^HuggingFaceTokenizer t id-seq {:keys [skip-special-tokens?] :or {skip-special-tokens? true}}]
   (.decode t (long-array id-seq) (boolean skip-special-tokens?))))

(defn build-sentence
  "Reconstruct a sentence from token strings with the tokenizer's native
  token-joining rules. This is different from a decode of token ids."
  [^HuggingFaceTokenizer t token-strings]
  (.buildSentence t ^java.util.List (vec token-strings)))

(defn native-version
  "Version of the loaded native tokenizers runtime."
  [^HuggingFaceTokenizer t]
  (.getVersion t))

(defn batch-encode
  "Encode many `texts` at once, returning a vector of `encode`-shaped maps.
  Accepts the same options as `encode`."
  ([^HuggingFaceTokenizer t texts]
   (mapv enc->map (.batchEncode t ^java.util.List (vec texts))))
  ([^HuggingFaceTokenizer t texts
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (mapv enc->map (.batchEncode t ^java.util.List (vec texts)
                               (boolean add-special-tokens?)
                               (boolean with-overflowing-tokens?)))))

(defn batch-count-tokens
  "Token-id counts for `texts` via native batch encoding."
  ([^HuggingFaceTokenizer t texts]
   (mapv #(alength (.getIds ^Encoding %))
         (.batchEncode t ^java.util.List (vec texts))))
  ([^HuggingFaceTokenizer t texts
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (mapv #(alength (.getIds ^Encoding %))
         (.batchEncode t ^java.util.List (vec texts)
                       (boolean add-special-tokens?)
                       (boolean with-overflowing-tokens?)))))

(defn batch-ids
  "Token ids for `texts` via one native batch encoding."
  ([^HuggingFaceTokenizer t texts]
   (mapv #(vec (.getIds ^Encoding %))
         (.batchEncode t ^java.util.List (vec texts))))
  ([^HuggingFaceTokenizer t texts
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (mapv #(vec (.getIds ^Encoding %))
         (.batchEncode t ^java.util.List (vec texts)
                       (boolean add-special-tokens?)
                       (boolean with-overflowing-tokens?)))))

(defn batch-encode->ndlist
  "Encode `texts` directly to one batched DJL `NDList` owned by `manager`.
  Opts include the `batch-encode` opts plus `:with-token-type-ids?` and
  `:int32?` (both default false). The caller owns and must close the
  `NDManager`."
  ([^HuggingFaceTokenizer t texts manager]
   (batch-encode->ndlist t texts manager {}))
  ([^HuggingFaceTokenizer t texts manager
    {:keys [add-special-tokens? with-overflowing-tokens?
            with-token-type-ids? int32?]
     :or {add-special-tokens? true
          with-overflowing-tokens? false
          with-token-type-ids? false
          int32? false}}]
   (Encoding/toNDList
    (.batchEncode t ^java.util.List (vec texts)
                  (boolean add-special-tokens?)
                  (boolean with-overflowing-tokens?))
    manager (boolean with-token-type-ids?) (boolean int32?))))

(defn batch-decode
  "Decode many id sequences. Opts: `:skip-special-tokens?` (default true)."
  ([^HuggingFaceTokenizer t id-seqs]
   (batch-decode t id-seqs {}))
  ([^HuggingFaceTokenizer t id-seqs
    {:keys [skip-special-tokens?] :or {skip-special-tokens? true}}]
   (vec (.batchDecode t ^"[[J" (into-array (map long-array id-seqs))
                      (boolean skip-special-tokens?)))))

(defn- ->pair-list [pairs]
  (let [pair-list (PairList.)]
    (doseq [[text text-pair] pairs]
      (.add pair-list text text-pair))
    pair-list))

(defn batch-encode-pairs
  "Encode `[text text-pair]` pairs, returning encode-shaped maps."
  ([^HuggingFaceTokenizer t pairs]
   (mapv enc->map (.batchEncode t ^PairList (->pair-list pairs))))
  ([^HuggingFaceTokenizer t pairs
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (mapv enc->map (.batchEncode t ^PairList (->pair-list pairs)
                               (boolean add-special-tokens?)
                               (boolean with-overflowing-tokens?)))))
