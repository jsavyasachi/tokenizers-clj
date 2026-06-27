(defproject net.clojars.savya/tokenizers-clj "0.1.1"
  :description "Idiomatic Clojure wrapper over DJL's HuggingFace tokenizers (native Rust tokenizers via JNI): encode, decode, and count tokens."
  :url "https://github.com/jsavyasachi/tokenizers-clj"
  :license {:name "Eclipse Public License 2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :scm {:name "git" :url "https://github.com/jsavyasachi/tokenizers-clj"}
  :dependencies [[ai.djl.huggingface/tokenizers "0.36.0"]]
  :global-vars {*warn-on-reflection* true}
  :profiles {:provided {:dependencies [[org.clojure/clojure "1.12.0"]]}
             :dev {:dependencies [[org.clojure/clojure "1.12.0"]]
                   :resource-paths ["test/resources"]}
             :clojure-1-10 {:dependencies [[org.clojure/clojure "1.10.3"]]}
             :clojure-1-11 {:dependencies [[org.clojure/clojure "1.11.4"]]}
             :clojure-1-12 {:dependencies [[org.clojure/clojure "1.12.0"]]}}
  :aliases {"all" ["with-profile" "+clojure-1-10:+clojure-1-11:+clojure-1-12"]}
  :deploy-repositories [["clojars" {:url "https://repo.clojars.org"
                                    :username :env/clojars_username
                                    :password :env/clojars_password
                                    :sign-releases false}]])
