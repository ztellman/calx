;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns 
  ^{:skip-wiki true}
  calx.data
  (:use
    [clojure.contrib.def :only (defmacro- defvar- defvar)]
    [clojure.walk]
    [calx.core]
    [cantor])
  (:import [com.nativelibs4java.opencl CLContext CLByteBuffer CLMem CLMem$Usage CLEvent]
	   [com.nativelibs4java.util NIOUtils]
	   [java.nio ByteOrder ByteBuffer]))

;;;

(defvar- usage-types
  {:in CLMem$Usage/Input
   :out CLMem$Usage/Output
   :in-out CLMem$Usage/InputOutput})

(defvar- put-fns
  {:float #(.putFloat ^ByteBuffer %1 %2)
   :double #(.putDouble ^ByteBuffer %1 %2)
   :int #(.putInt ^ByteBuffer %1 %2)
   :long #(.putLong ^ByteBuffer %1 %2)
   :short #(.putShort ^ByteBuffer %1 %2)
   :byte #(.put ^ByteBuffer %1 (byte %2))})

(defvar- get-fns
  {:float #(.getFloat ^ByteBuffer %)
   :double #(.getDouble ^ByteBuffer %)
   :int #(.getInt ^ByteBuffer %)
   :long #(.getLong ^ByteBuffer %)
   :short #(.getShort ^ByteBuffer %)
   :byte #(.get ^ByteBuffer %)})

(defprotocol Signature
  (num-bytes [s])
  (flattened [s])
  (signature [s])
  (get-element [s b]))

(defn- signature? [s]
  (= (->> s meta :type) ::signature))

(defvar- type?
  #{:float :double :byte :short :int :long})

(defn- create-signature [s]
  (if (signature? s)
    s
    (let [wrapped (if (sequential? s) s [s])
	  flattened (flatten (postwalk #(if (map? %) (vals %) %) wrapped))
	  num-bytes (reduce +
		      (map
			{:float 4
			 :double 8
			 :int 4
			 :long 8
			 :short 2
			 :byte 1}
			flattened))]
      ^{:type ::signature}
      (reify Signature
	(num-bytes [_] num-bytes)
	(flattened [_] flattened)
	(signature [_] s)
	(get-element [_ buf]
	  (postwalk
	    #(if (type? %)
	       ((get-fns %) buf)
	       %)
	    s))))))

;;;

(defn to-buffer
  "Fills a ByteBuffer with a contiguous mixed datatype defined by 'sig'.  's' is assumed to be flat."
  [s sig]
  (let [sig (create-signature sig)
	buffer-size (* (/ (count s) (count (flattened sig))) (num-bytes sig))
	buf (NIOUtils/directBytes buffer-size (ByteOrder/nativeOrder))]
    (doseq [[typ val] (map list (cycle (flattened sig)) s)]
      ((put-fns typ) buf val))
    (.rewind ^ByteBuffer buf)
    buf))

(defn- element-from-buffer [^ByteBuffer buf sig idx]
  (let [buf ^ByteBuffer (.asReadOnlyBuffer buf)]
    (.order buf (ByteOrder/nativeOrder))
    (.position buf (* (num-bytes sig) idx))
    (get-element sig buf)))

(defn from-buffer
  "Pulls out mixed datatypes from a ByteBuffer, per 'sig'."
  [^ByteBuffer buf sig]
  (let [sig (create-signature sig)
	cnt (/ (.capacity buf) (num-bytes sig))]
    ^{:type ::from-buffer}
    (reify
      clojure.lang.Indexed
      (nth [_ i] (element-from-buffer buf sig i))
      (count [_] cnt)
      clojure.lang.Sequential
      clojure.lang.Seqable
      (seq [this] (map #(nth this %) (range cnt))))))

;;;

(defprotocol Data
  (mimic [d] [d elements] [d elements usage]
    "Create a different buffer of the same type. 'elements' and 'usage' default to those of the original buffer.")
  (enqueue-read [d] [d range]
    "Asynchronously copies a subset of the buffer into local memory. 'range' defaults to the full buffer.

     Returns an object that, when dereferenced, halts execution until the copy is complete, then returns a seq.")
  (enqueue-overwrite [destination destination-range source]
    "Asynchronously copies a buffer from local memory onto the given subset of the buffer.")
  (enqueue-copy [destination destination-offset source source-range]
    "Enqueues a copy from a subset of the source onto the destination.")
  (suitability [a size]
    "Returns the suitability for containing a data set of the specified size.")
  (released? [d]
    "Returns true if ref-count is equal to zero.")
  (acquire! [d]
    "Acquires the buffer.")
  (release! [d]
    "Releases the buffer."))

(declare create-buffer-)

(defn- best-match [cache num-bytes usage]
  (->> @cache
    (filter released?)
    (filter #(= usage (:usage %)))
    (map #(vector (suitability %1 num-bytes) %1))
    (filter first)
    (sort #(compare (second %) (first %)))
    first
    second))

(defn- find-match [cache num-bytes usage]
  (dosync
    (when-let [match (best-match cache num-bytes usage)]
      (acquire! match)
      match)))

(defn create-buffer
  "Creates an OpenCL buffer.

   'usage' may be one of [:in :out :in-out].  The default value is :in-out."
  ([elements sig]
     (create-buffer elements sig :in-out))
  ([elements sig usage]
     (let [sig (create-signature sig)]
       (if-let [match (find-match (cache) (num-bytes sig) usage)]
	 (assoc match
	   :elements elements
	   :signature sig)
	 (let [buffer (create-buffer-
			(.createByteBuffer (context) (usage-types usage) (* elements (num-bytes sig)))
			elements
			sig
			usage)]
	   (dosync (alter (cache) conj buffer))
	   buffer)))))

(defrecord Buffer [^CLByteBuffer buffer, ^int capacity, ^int elements, signature, usage, ref-count]
  Data
  ;;
  (mimic [this] (mimic this elements))
  (mimic [this elements] (mimic this elements usage))
  (mimic [_ elements usage] (create-buffer elements signature usage))
  ;;
  (acquire! [_] (dosync (alter ref-count inc)))
  (release! [_] (dosync (alter ref-count dec)))
  (released? [_] (zero? @ref-count))
  ;;
  (suitability [this size]
    (when (and (<= size capacity) (> size (/ capacity 2)))
      (/ (float size) capacity)))
  ;;
  (enqueue-read [this] (enqueue-read this (interval 0 elements)))
  (enqueue-read [this rng]
    (when-not (range? rng)
      (throw (Exception. "'rng' must be an interval, created via (cantor/interval upper lower)")))
    (let [elements (size rng)
	  buf (NIOUtils/directBytes (* elements (num-bytes signature)) (ByteOrder/nativeOrder))
	  event (. buffer read
		  (queue)
		  (* elements (num-bytes signature) (upper rng))
		  (* elements (num-bytes signature))
		  buf
		  false
		  (make-array CLEvent 0))]
      ^{:type ::enqueued-read}
      (reify
	HasEvent
	(event [_] event)
	(description [_] :enqueued-read)
	clojure.lang.IDeref
	(deref [_]
	  (wait-for event)
	  (from-buffer buf signature)))))
  ;;
  (enqueue-overwrite [_ dst-range src]
    (. buffer writeBytes
      (queue)
      (* (num-bytes signature) (upper dst-range))
      (* (num-bytes signature) (size dst-range))
      src
      true
      (make-array CLEvent 0)))
  ;;
  (enqueue-copy [_ dst-offset src src-range]
    (. (:buffer src) copyTo
      (queue)
      (* (num-bytes signature) (upper src-range))
      (* (num-bytes signature) (size src-range))
      buffer
      (* (num-bytes signature) dst-offset)
      (make-array CLEvent 0))))

(defn- create-buffer- [^CLByteBuffer buffer elements sig usage]
  (let [sig (create-signature sig)
	buf (Buffer. buffer (* elements (num-bytes sig)) elements sig usage (ref 1))]
    (with-meta buf (merge (meta buf) {:cl-object buffer}))))

(defn wrap
  "Copies a sequence into an OpenCL buffer.

   'usage' may be one of [:in :out :in-out].  The default value is :in-out."
  ([s]
     (wrap s 1))
  ([s sig]
     (wrap s sig :in-out))
  ([s sig usage]
     (let [sig (create-signature sig)
	   s (flatten s)
	   num-bytes (* (num-bytes sig) (/ (count s) (count (flattened sig))))
	   sig-count (count (flattened sig))]
       (when-not (zero? (rem (count s) sig-count))
	 (throw (Exception. (format "Sequence count (%d) not evenly divisable by signature count (%d)." (count s) sig-count))))
       (create-buffer-
	 (.createByteBuffer (context) (usage-types usage) (to-buffer s sig) false)
	 (/ (count s) sig-count)
	 sig
	 usage))))

;;;

(defmethod print-method ::enqueued-read [b writer]
  (let [sts (status (event b))]
    (.write writer "#<Ref to native buffer: ")
    (if (= :complete sts)
      (print-method (seq @b) writer)
      (.write writer sts))
    (.write writer ">")))

(defmethod print-method ::from-buffer [b writer]
  (print-method (seq b) writer))
