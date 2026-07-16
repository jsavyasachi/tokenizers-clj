(ns tokenizers.core
  "Idiomatic Clojure wrapper over DJL's HuggingFace tokenizers, which bind the native
  Rust `tokenizers` library via JNI. Build a tokenizer with `from-file` /
  `from-pretrained` / `from-stream`, then `encode`, `decode`, or `count-tokens`.

  A tokenizer holds a native handle: close it (`with-open` works) to free it."
  (:require [clojure.string :as str])
  (:import [ai.djl.huggingface.tokenizers HuggingFaceTokenizer Encoding TokenizerConfig]
           [ai.djl.huggingface.tokenizers.jni CharSpan]
           [java.io File InputStream]
           [java.net URI URLEncoder]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest
            HttpResponse$BodyHandlers]
           [java.nio.charset StandardCharsets]
           [java.nio.file CopyOption Files LinkOption OpenOption Path StandardCopyOption]
           [java.nio.file.attribute FileAttribute]
           [java.util Locale]
           [ai.djl.util PairList]))

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

(defn- encode-component [value]
  (.replace (URLEncoder/encode (str value) StandardCharsets/UTF_8) "+" "%20"))

(defn- hub-uri [id revision]
  (URI/create
   (str "https://huggingface.co/"
        (str/replace (encode-component id) "%2F" "/")
        "/resolve/" (encode-component revision) "/tokenizer.json")))

(defn- hub-cache-path [id revision cache-dir]
  (-> (as-path (or cache-dir
                   (str (System/getProperty "user.home")
                        File/separator ".cache" File/separator
                        "huggingface" File/separator "tokenizers-clj")))
      (.resolve ^String (encode-component id))
      (.resolve ^String (encode-component revision))
      (.resolve "tokenizer.json")))

(defn- download-tokenizer! [uri ^Path target auth-token]
  (Files/createDirectories (.getParent target) (make-array FileAttribute 0))
  (let [request-builder (doto (HttpRequest/newBuilder uri) (.GET))
        _ (when auth-token
            (.header request-builder "Authorization" (str "Bearer " auth-token)))
        client (-> (HttpClient/newBuilder)
                   (.followRedirects HttpClient$Redirect/ALWAYS)
                   (.build))
        response (.send client (.build request-builder)
                        (HttpResponse$BodyHandlers/ofByteArray))
        status (.statusCode response)]
    (when-not (<= 200 status 299)
      (throw (ex-info (str "HuggingFace Hub returned HTTP " status " for " uri)
                      {:status status :uri (str uri)})))
    (let [temp (Files/createTempFile (.getParent target) ".tokenizer-" ".json"
                                     (make-array FileAttribute 0))]
      (try
        (Files/write temp ^bytes (.body response) (make-array OpenOption 0))
        (Files/move temp target
                    (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))
        (finally
          (Files/deleteIfExists temp))))
    target))

(def ^:private hub-option-keys
  #{:revision :auth-token :cache-dir :local-only? :local-only :offline? :offline})

(defn- wrapper-managed-hub? [opts]
  (some #(contains? opts %) [:revision :cache-dir :local-only? :local-only
                             :offline? :offline]))

(defn- offline? [opts]
  (boolean (or (:local-only? opts) (:local-only opts)
               (:offline? opts) (:offline opts))))

(defn from-pretrained
  "Tokenizer by HuggingFace Hub id. Options include `:revision`, `:auth-token`,
  `:cache-dir`, and `:local-only?` / `:offline?`, plus constructor options."
  (^HuggingFaceTokenizer [^String id]
   (from-pretrained id {}))
  (^HuggingFaceTokenizer [^String id opts]
   (assert-compatible-native-runtime!)
   (if (wrapper-managed-hub? opts)
     (let [revision (str (or (:revision opts) "main"))
           path (hub-cache-path id revision (:cache-dir opts))]
       (when-not (Files/exists path (make-array LinkOption 0))
         (if (offline? opts)
           (throw (ex-info (str "Tokenizer " id " at revision " revision
                               " was not found in the local cache")
                           {:id id :revision revision :cache-path (str path)}))
           (download-tokenizer! (hub-uri id revision) path (:auth-token opts))))
       (from-file path (apply dissoc opts hub-option-keys)))
     (HuggingFaceTokenizer/newInstance
      id
      (cond-> (djl-options opts)
        (:auth-token opts) (assoc "hf_token" (str (:auth-token opts))))))))

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
  "Encode `text`, optionally paired with a second text, into a token-data map.
  Opts: `:add-special-tokens?` (default true), `:with-overflowing-tokens?`
  (default false)."
  ([^HuggingFaceTokenizer t ^String text]
   (enc->map (.encode t text)))
  ([^HuggingFaceTokenizer t ^String text pair-or-opts]
   (if (map? pair-or-opts)
     (let [{:keys [add-special-tokens? with-overflowing-tokens?]
            :or {add-special-tokens? true with-overflowing-tokens? false}} pair-or-opts]
       (enc->map (.encode t text (boolean add-special-tokens?)
                          (boolean with-overflowing-tokens?))))
     (enc->map (.encode t text ^String pair-or-opts))))
  ([^HuggingFaceTokenizer t ^String text ^String text-pair
    {:keys [add-special-tokens? with-overflowing-tokens?]
     :or {add-special-tokens? true with-overflowing-tokens? false}}]
   (enc->map (.encode t text text-pair (boolean add-special-tokens?)
                      (boolean with-overflowing-tokens?)))))

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
