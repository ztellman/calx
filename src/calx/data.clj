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
  (:use [clojure.contrib.def :only (defmacro- defvar- defvar)]
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

(defn- transform-sig [sig]
  (let [sig (if (map? sig) (vals sig) sig)
	sig (if (sequential? sig) sig [sig])]
    sig))

(defn- sizeof [sig]
  (reduce +
    (map
      {:float 4
       :double 8
       :int 4
       :long 8
       :short 2
       :byte 1}
      (transform-sig sig))))

(defn to-buffer
  "Fills a ByteBuffer with a contiguous mixed datatype defined by 'sig'."
  [s sig]
  (let [sig (transform-sig sig)
	num-bytes (* (/ (count s) (count sig)) (sizeof sig))
	buf (NIOUtils/directBytes num-bytes (ByteOrder/nativeOrder))]
    (doseq [[typ val] (map list (cycle sig) s)]
      ((put-fns typ) buf val))
    (.rewind ^ByteBuffer buf)
    buf))

(defn- element-from-buffer [^ByteBuffer buf sig idx]
  (.position buf (* (sizeof sig) idx))
  (let [transformed-sig (transform-sig sig)
	element (doall (map #((get-fns %1) buf) (transform-sig sig)))]
    (.rewind buf)
    (if (and (not (map? sig)) (= 1 (count transformed-sig)))
      (first element)
      (if (map? sig)
	(zipmap (keys sig) element)
	element))))

(defn from-buffer
  "Pulls out mixed datatypes from a ByteBuffer, per 'sig'."
  [^ByteBuffer buf sig]
  (let [cnt (/ (.capacity buf) (sizeof sig))]
    ^{:type ::from-buffer}
    (reify
      clojure.lang.Indexed
      (nth [_ i] (element-from-buffer buf sig i))
      (count [_] cnt)
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
     (let [num-bytes (* elements (sizeof sig))]
       (if-let [match (find-match (cache) num-bytes usage)]
	 (assoc match
	   :elements elements
	   :signature sig)
	 (let [buffer (create-buffer-
			(.createByteBuffer (context) (usage-types usage) (* elements (sizeof sig)))
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
	  buf (NIOUtils/directBytes (* elements (sizeof signature)) (ByteOrder/nativeOrder))
	  event (. buffer read
		  (queue)
		  (* elements (sizeof signature) (upper rng))
		  (* elements (sizeof signature))
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
    (let [sig-size (sizeof signature)]
      (. buffer writeBytes
	(queue)
	(* sig-size (upper dst-range))
	(* sig-size (size dst-range))
	src
	true
	(make-array CLEvent 0))))
  ;;
  (enqueue-copy [_ dst-offset src src-range]
    (let [sig-size (sizeof signature)]
      (. (:buffer src) copyTo
	(queue)
	(* sig-size (upper src-range))
	(* sig-size (size src-range))
	buffer
	(* sig-size dst-offset)
	(make-array CLEvent 0)))))

(defn- create-buffer- [^CLByteBuffer buffer elements sig usage]
  (let [buf (Buffer. buffer (* elements (sizeof sig)) elements sig usage (ref 1))]
    (with-meta buf (merge (meta buf) {:cl-object buffer}))))

(defn wrap
  "Copies a sequence into an OpenCL buffer.  Type is assumed to be uniform across the sequence.

   'usage' may be one of [:in :out :in-out].  The default value is :in-out."
  ([s]
     (wrap s 1))
  ([s sig]
     (wrap s sig :in-out))
  ([s sig usage]
     (let [sig-count (if (sequential? sig) (count sig) 1)
	   num-bytes (* (sizeof sig) (/ (count s) sig-count))]
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
