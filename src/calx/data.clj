;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns calx.data
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

(defn sizeof [sig]
  (let [sig (if (sequential? sig) sig [sig])]
    (reduce +
      (map
	{:float 4
	 :double 8
	 :int 4
	 :long 8
	 :short 2
	 :byte 1}
	sig))))

(defn to-buffer [s sig]
  (let [sig (if (sequential? sig) sig [sig])
	dim (* (/ (count s) (count sig)) (sizeof sig))
	buf (NIOUtils/directBytes dim (ByteOrder/nativeOrder))]
    (doseq [[typ val] (map list (cycle sig) s)]
      ((put-fns typ) buf val))
    (.rewind ^ByteBuffer buf)
    buf))

(defn- element-from-buffer [^ByteBuffer buf sig idx]
  (.position buf (* (sizeof sig) idx))
  (let [element (vec (map #((get-fns %1) buf) sig))]
    (.rewind buf)
    (if (= 1 (count sig))
      (first element)
      element)))

(defn from-buffer [^ByteBuffer buf sig]
  (let [sig (if (sequential? sig) sig [sig])
	cnt (/ (.capacity buf) (sizeof sig))]
    ^{:type ::from-buffer}
    (reify
      clojure.lang.Indexed
      (nth [_ i] (element-from-buffer buf sig i))
      (count [_] cnt)
      clojure.lang.Seqable
      (seq [this] (map #(nth this %) (range cnt))))))

;;;

(defprotocol Data
  (mimic [d] [d dim usage]
    "Create a different buffer of the same type. 'dim' and 'usage' default to those of the original buffer.")
  (enqueue-read [d] [d range]
    "Asynchronously copies a subset of the buffer into local memory. 'range' defaults to the full buffer.
     Returns an object that, when dereferenced, halts execution until the copy is complete, then returns a seq.")
  (dim [d]
    "Returns the dimensions of the buffer.")
  (signature [d]
    "Returns the per-element signature of the buffer.")
  (release [d]
    "Releases the buffer."))

(defn- create-buffer- [^CLByteBuffer cl-buf sig dim usage]
  (let [element-bytes (sizeof sig)]
    ^{:cl-object cl-buf}
    (reify
      Data
      (mimic [this] (mimic this dim usage))
      (mimic [_ dim usage]
	(create-buffer-
	  (.createByteBuffer (context) (usage-types usage) (* dim element-bytes))
	  sig
	  dim
	  usage))
      (signature [_] sig)
      (dim [_] dim)
      (enqueue-read [this] (enqueue-read this (interval 0 dim)))
      (enqueue-read [this rng]
	(let [dim (size rng)
	      buf (NIOUtils/directBytes (* dim element-bytes) (ByteOrder/nativeOrder))
	      event (. cl-buf read
		      (queue)
		      (* element-bytes (ul rng))
		      (* element-bytes (size rng))
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
	      (from-buffer buf sig)))))
      (release [_] (.release cl-buf)))))

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
	 sig
	 (/ (count s) sig-count)
	 usage))))

(defn create-buffer
  ([dim sig]
     (create-buffer dim sig :in-out))
  ([dim sig usage]
     (create-buffer-
       (.createByteBuffer (context) (usage-types usage) (* dim (sizeof sig)))
       sig
       dim
       usage)))

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
