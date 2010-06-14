(ns calx.data
  (:use [clojure.contrib.def :only (defmacro- defvar- defvar)]
	[calx.core]
	[cantor])
  (:import [com.nativelibs4java.opencl CLContext CLBuffer CLMem CLMem$Usage CLEvent]
	   [com.nativelibs4java.util NIOUtils]
	   [java.nio ByteOrder Buffer]))

;;;

(defvar- usage-types
  {:input CLMem$Usage/Input
   :output CLMem$Usage/Output
   :input-output CLMem$Usage/InputOutput})

(defn- buffer-type [typ]
  (symbol (str "java.nio." typ "Buffer")))

(defmacro- direct-buffer-fn [typ]
  `(fn [dim#]
     (. NIOUtils ~(symbol (str "direct" typ "s")) dim# (ByteOrder/nativeOrder))))

(defmacro- populate-buffer-fn [typ]
  `(fn [buf# seq#]
     (doseq [s# seq#]
       (.put ^{:tag ~(buffer-type typ)} buf# s#))
     (.rewind buf#)))

(defmacro- create-cl-buffer-fn [typ]
  (let [fn-name (symbol (str "create" typ "Buffer"))] 
    `(fn [buf# usage#]
       (. (context) ~fn-name (usage-types usage#) buf# false))))

(defmacro- wrap-seq-fn [typ]
  `(fn [s# usage#]
     (let [buf# ((direct-buffer-fn ~typ) (count s#))]
       ((populate-buffer-fn ~typ) buf# s#)
       ((create-cl-buffer-fn ~typ) buf# usage#))))

(defmacro- empty-buffer-fn [typ]
  `(fn [dim# usage#]
     (let [buf# ((direct-buffer-fn ~typ) dim#)]
       ((create-cl-buffer-fn ~typ) buf# usage#))))

(defmacro- buffer-nth-fn [typ]
  (fn [buf# idx#]
    (.get ^{:tag ~(buffer-type typ)} buf# idx#)))

(defvar type-map
  (->>
   {[:int Integer Integer/TYPE] 'Int
    [:short Short Short/TYPE] 'Short
    [:byte Byte Byte/TYPE] 'Byte
    [:long Long Long/TYPE] 'Long
    [:float Float Float/TYPE] 'Float
    [:double Double Double/TYPE] 'Double}
   (map (fn [[k v]] (interleave k (repeat v))))
   (apply concat)
   (apply hash-map)))

(defmacro apply-type-fn [type-macro]
  (into {} (map (fn [[k v]] [k (macroexpand (list type-macro v))]) type-map)))

(defvar wrap-fns (apply-type-fn wrap-seq-fn))
(defvar create-fns (apply-type-fn empty-buffer-fn))
(defvar buffer-fns (apply-type-fn direct-buffer-fn))
(defvar nth-fns (apply-type-fn buffer-nth-fn))

;;;

(defprotocol Data
  (mimic [d] [d dim])
  (enqueue-read [d] [d range])
  (release [d]))

(defn- create- [buf typ dim usage]
  (reify
   Data
   (mimic [this] (mimic this dim))
   (mimic [_ dim] ((create-fns typ) dim))
   (enqueue-read [this] (enqueue-read this (interval 0 dim)))
   (enqueue-read
    [this rng]
    (let [dim (size rng)
	  read-buf ((buffer-fns typ) dim)
	  event (. buf read
		   (queue)
		   (ul rng) (size rng)
		   read-buf false
		   (make-array CLEvent 0))
	  nth-fn (nth-fns typ)]
      (reify
       HasEvent
       (event [_] event)
       clojure.lang.IDeref
       (deref [_]
         (wait-for event)
	 (reify
	  clojure.lang.Indexed
	  (nth [_ i] (nth-fn read-buf i))
	  (count [_] dim)
	  clojure.lang.Seqable
	  (seq [this] (map #(nth this %) (range (count this)))))))))
   (release [_] (.release buf))))

(defn create
  ([typ dim]
     (create typ dim :input-output))
  ([typ dim usage]
     (let [buf #^CLBuffer ((create-fns typ) dim)]
       (create- buf typ dim usage))))

(defn wrap
  ([s]
     (wrap s :input-output))
  ([s usage]
     (let [typ (->> s first type)
	   buf #^CLBuffer ((wrap-fns typ) s usage)]
       (create- buf typ (count s) usage))))

