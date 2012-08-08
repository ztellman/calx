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
    [gloss core io]
    [calx.core])
  (:import [com.nativelibs4java.opencl CLContext CLByteBuffer CLMem CLMem$Usage CLEvent]
	   [com.nativelibs4java.util NIOUtils]
	   [java.nio ByteOrder ByteBuffer]))

;;;

(def usage-types
  {:in CLMem$Usage/Input
   :out CLMem$Usage/Output
   :in-out CLMem$Usage/InputOutput})

;;;

(defn create-byte-buffer
  ([size]
     (create-byte-buffer
       (or
	 (when *context*
	   (.getByteOrder ^CLContext (:context *context*)))
	 (ByteOrder/nativeOrder))
       size))
  ([byte-order size]
     (NIOUtils/directBytes size byte-order)))

(defn to-buffer
  [s frame]
  (let [codec (compile-frame frame)
	buffer-size (* (sizeof codec) (count s))
	buf ^ByteBuffer (create-byte-buffer buffer-size)]
    (encode-to-buffer codec buf s)
    (.rewind buf)))

(defn from-buffer
  "Pulls out mixed datatypes from a ByteBuffer."
  [^ByteBuffer buf frame]
  (decode-all (compile-frame frame) buf))

;;;

(defprotocol Data
  (mimic [d] [d elements] [d elements usage]
    "Create a different buffer of the same type. 'elements' and 'usage' default to those of the original buffer.")
  (enqueue-read [d] [d range]
    "Asynchronously copies a subset of the buffer into local memory. 'range' defaults to the full buffer.

     Returns an object that, when dereferenced, halts execution until the copy is complete, then returns a seq.")
  (enqueue-overwrite [destination [lower upper] source]
    "Asynchronously copies a buffer from local memory onto the given subset of the buffer.")
  (enqueue-copy [destination destination-offset source [lower upper]]
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
  ([elements frame]
     (create-buffer elements frame :in-out))
  ([elements frame usage]
     (let [codec (compile-frame frame)]
       (if-let [match (find-match (cache) (sizeof codec) usage)]
	 (assoc match
	   :elements elements
	   :codec codec)
	 (let [buffer (create-buffer-
			(.createByteBuffer (context) (usage-types usage) (* elements (sizeof codec)))
			elements
			frame
			usage)]
	   (dosync (alter (cache) conj buffer))
	   buffer)))))

(defrecord Buffer [^CLByteBuffer buffer, ^int capacity, ^int elements, codec, usage, ref-count]
  Data
  ;;
  (mimic [this] (mimic this elements))
  (mimic [this elements] (mimic this elements usage))
  (mimic [_ elements usage] (create-buffer elements codec usage))
  ;;
  (acquire! [_] (dosync (alter ref-count inc)))
  (release! [_] (dosync (alter ref-count dec)))
  (released? [_] (zero? @ref-count))
  ;;
  (suitability [this size]
    (when (and (<= size capacity) (> size (/ capacity 2)))
      (/ (float size) capacity)))
  ;;
  (enqueue-read [this] (enqueue-read this [0 elements]))
  (enqueue-read [this [lower upper]]
    (let [elements (- upper lower)
	  buf (create-byte-buffer (* elements (sizeof codec)))
	  event (. buffer read
		  (queue)
		  (* lower (sizeof codec))
		  (* elements (sizeof codec))
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
	  (from-buffer buf codec)))))
  ;;
  (enqueue-overwrite [_ [lower upper] src]
    (. buffer writeBytes
      (queue)
      (* (sizeof codec) upper)
      (* (sizeof codec) (- upper lower))
      src
      true
      (make-array CLEvent 0)))
  ;;
  (enqueue-copy [_ offset src [lower upper]]
    (. (:buffer src) copyTo
      (queue)
      (* (sizeof codec) upper)
      (* (sizeof codec) (- upper lower))
      buffer
      (* (sizeof codec) offset)
      (make-array CLEvent 0))))

(defn- create-buffer- [^CLByteBuffer buffer elements frame usage]
  (let [codec (compile-frame frame)
	buf (Buffer. buffer (* elements (sizeof codec)) elements codec usage (ref 1))]
    (with-meta buf (merge (meta buf) {:cl-object buffer}))))

(defn wrap
  "Copies a sequence into an OpenCL buffer.

   'usage' may be one of [:in :out :in-out].  The default value is :in-out."
  ([s]
     (wrap s :byte))
  ([s frame]
     (wrap s frame :in-out))
  ([s frame usage]
     (let [codec (compile-frame frame)]
       (create-buffer-
	 (.createByteBuffer (context) (usage-types usage) (to-buffer s codec) false)
	 (count s)
	 codec
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
