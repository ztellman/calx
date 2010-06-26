;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns 
  ^{:author "Zachary Tellman"
    :doc "An idiomatic wrapper for OpenCL."}
  calx
  (:use
    [clojure.contrib.def :only (defmacro- defvar-)])
  (:require
    [calx
     [core :as core]
     [data :as data]])
  (:import
    [com.nativelibs4java.opencl
     CLEvent CLQueue CLEvent$CommandType CLContext]))

;;;

(defmacro- import-fn [sym]
  (let [m (meta (eval sym))
        m (meta (intern (:ns m) (:name m)))
        n (:name m)
        arglists (:arglists m)
        doc (:doc m)]
    (list `def (with-meta n {:doc doc :arglists (list 'quote arglists)}) (eval sym))))

(import-fn core/available-platforms)
(import-fn core/available-devices)
(import-fn core/available-cpu-devices)
(import-fn core/available-gpu-devices)
(import-fn core/best-device)
(import-fn core/version)

(import-fn core/platform)
(import-fn core/context)
(import-fn core/queue)
(import-fn core/program)

(import-fn core/create-context)
(import-fn core/create-queue)
(import-fn core/finish)
(import-fn core/enqueue-marker)
(import-fn core/enqueue-barrier)
(import-fn core/enqueue-wait-for)
(import-fn core/wait-for)
(import-fn core/status)

(import-fn core/compile-program)
(import-fn core/local)
(import-fn core/enqueue-kernel)

(import-fn data/to-buffer)
(import-fn data/from-buffer)
(import-fn data/wrap)
(import-fn #'data/mimic)
(import-fn #'data/release!)
(import-fn #'data/acquire!)
(import-fn #'data/enqueue-read)
(import-fn data/create-buffer)

;;;

(defmacro with-queue
  "Executes inner scope within the queue."
  [q & body]
  `(binding [core/*queue* ~q]
     ~@body))

(defmacro with-queue-and-wait
  "Executes inner scope within the queue, and waits for all commands to complete."
  [q & body]
  `(with-queue ~q
     (try
       ~@body
       (finally
	 (finish)))))

(defmacro with-program [program & body]
  `(binding [core/*program* ~program]
     ~@body))

(defmacro with-platform
  "Defines the platform within the inner scope."
  [platform & body]
  `(binding [core/*platform* ~platform]
     ~@body))

(defmacro with-context
  "Defines the context within the inner scope."
  [context & body]
  `(let [context# ~context] 
     (with-platform (.getPlatform ^CLContext (:context context#))
       (binding [core/*context* context#]
	 (with-queue-and-wait (create-queue)
	   ~@body)))))

(defmacro with-devices
  "Defines the devices within the inner scope.  Creates a context using these devices, and releases the context once the scope is exited."
  [devices & body]
  `(with-platform nil
     (let [context# (apply create-context ~devices)]
       (try
	 (with-context context#
	   ~@body)
	 (finally
	   (.release ^CLContext (:context context#)))))))

(defmacro with-cpu
  "Executes the inner scope inside a context using the CPU."
  [& body]
  `(with-devices [(first (available-cpu-devices))] ~@body))

(defmacro with-gpu
  "Executes the inner scope inside a context using the GPU."
  [& body]
  `(with-devices [(first (available-gpu-devices))] ~@body))

(defmacro with-cl
  "Executes the inner scope inside a context using the best available device."
  [& body]
  `(with-devices [(best-device)] ~@body))

;;;

(defvar- event-type-map
  {CLEvent$CommandType/CopyBuffer :copy-buffer
   CLEvent$CommandType/CopyBufferToImage :copy-buffer-to-image
   CLEvent$CommandType/CopyImageToBuffer :copy-image-to-buffer
   CLEvent$CommandType/ReadBuffer :read-buffer
   CLEvent$CommandType/WriteBuffer :write-buffer
   CLEvent$CommandType/ReadImage :copy-image
   CLEvent$CommandType/WriteImage :write-image
   CLEvent$CommandType/NDRangeKernel :execute-kernel})

(extend-type CLEvent
  core/HasEvent
  (event [e]
    e)
  (description [e]
    (or
      (event-type-map (.getCommandType e))
      :other)))

(extend-type CLQueue
  core/HasEvent
  (event [q]
    (with-queue q
      (enqueue-marker)))
  (description [q]
    :queue))

;;;
