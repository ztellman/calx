;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns calx.core
  (:use
    [clojure.contrib.def :only (defvar defvar-)]
    [clojure.contrib.seq :only (indexed)]
    [cantor])
  (:import
    [com.nativelibs4java.opencl
     JavaCL CLContext CLPlatform
     CLDevice CLDevice$QueueProperties 
     CLQueue CLEvent CLEvent$CommandExecutionStatus CLEvent$CommandType
     CLProgram CLKernel CLKernel$LocalSize]))

;;;

(defn available-platforms
  "Returns a list of all available platforms."
  []
  (seq (JavaCL/listPlatforms)))

;;;

(defvar *platform* nil "The current platform.")
(defvar *context* nil "The current context.")
(defvar *queue* nil "The current queue.")
(defvar *program* nil "The current program")
(defvar *workgroup-size* nil "The size of the workgroup")

(defn platform
  "Returns the current platform, or throws an exception if it's not defined."
  []
  (or *platform* (first (available-platforms))))

(defn context
  "Returns the current context, or throws an exception if it's not defined."
  []
  (when-not *context*
    (throw (Exception. "OpenCL context not defined within this scope.")))
  *context*)

(defn devices
  "Returns the devices of the current context, or an exception if they're not defined."
  []
  (seq (.getDevices (context))))

(defn queue
  "Returns the current queue, or throws an exception if it's not defined."
  []
  (when-not *queue*
    (throw (Exception. "OpenCL queue not defined within this scope.")))
  *queue*)

(defn program
  "Returns the current program, or throws an exception if it's not defined."
  []
  (when-not *program*
    (throw (Exception. "OpenCL program not defined within this scope.")))
  *program*)

;;;

(defn available-devices
  "Returns all available devices."
  ([]
     (available-devices (platform)))
  ([platform]
     (seq (.listAllDevices ^CLPlatform platform true))))

(defn available-cpu-devices
  "Returns all available CPU devices."
  ([]
     (available-cpu-devices (platform)))
  ([platform]
     (seq (.listCPUDevices ^CLPlatform platform true))))

(defn available-gpu-devices
  "Returns all available GPU devices."
  ([]
     (available-gpu-devices (platform)))
  ([platform]
     (seq (.listGPUDevices ^CLPlatform platform true))))

(defn best-device
  "Returns the best available device."
  ([]
     (best-device (platform)))
  ([platform]
     (.getBestDevice ^CLPlatform platform)))

(defn version
  "Returns the version of the OpenCL implementation."
  ([]
     (version (platform)))
  ([platform]
     (.getVersion ^CLPlatform platform)))

;;;

(defprotocol HasEvent
  (event [e] "Returns the event which represents the completion of this object.")
  (description [e] "Describes the event."))

(defn create-queue
  "Creates a queue."
  ([]
     (create-queue (first (devices))))
  ([^CLDevice device & properties]
     (.createQueue
       device
       (context)
       (if (empty? properties)
	 (make-array CLDevice$QueueProperties 0)
	 (into-array properties)))))

(defmacro with-queue
  "Executes inner scope within the queue."
  [q & body]
  `(binding [*queue* ~q]
     ~@body))

(defmacro enqueue
  "Executes inner scope within a default queue."
  [& body]
  `(with-queue (create-queue)
     ~@body))

(defmacro enqueue-and-wait
  "Executes inner scope within a default queue, and waits for all commands to complete."
  [& body]
  `(enqueue
     (try
       ~@body
       (finally
	 (finish)))))

(defn finish
  "Halt execution until all enqueued operations are complete."
  []
  (.finish ^CLQueue (queue)))

(defn enqueue-barrier
  "Ensures that all previously enqueued commands will complete before new commands will begin."
  []
  (.enqueueBarrier ^CLQueue (queue)))

(defn enqueue-marker
  "Returns an event which represents the progress of all previously enqueued commands."
  [q]
  (.enqueueMarker ^CLQueue (queue)))

(defn enqueue-wait
  "Enqueues a barrier which will halt execution until all events have completed."
  [& events]
  (.enqueueWaitForEvents ^CLQueue (queue) (into-array (map event events))))

;;;

(defn wait-for
  "Halt execution until all events are complete."
  [& events]
  (CLEvent/waitFor (into-array (map event events))))

(defn status
  "Returns the status of the event."
  [^CLEvent event]
  ({CLEvent$CommandExecutionStatus/Complete :complete
    CLEvent$CommandExecutionStatus/Queued :enqueued
    CLEvent$CommandExecutionStatus/Running :running
    CLEvent$CommandExecutionStatus/Submitted :submitted}
   (.getCommandExecutionStatus event)))

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
  HasEvent
  (event [e]
    e)
  (description [e]
    (or
      (event-type-map (.getCommandType e))
      :other)))

(extend-type CLQueue
  HasEvent
  (event [q]
    (with-queue q
      (enqueue-marker)))
  (description [q]
    :queue))

;;;

(defmacro with-platform
  "Defines the platform within the inner scope."
  [platform & body]
  `(binding [*platform* ~platform]
     ~@body))

(defn create-context
  "Creates a context which uses the specified devices.  Using more than one device is not recommended."
  [& devices]
  (.createContext ^CLPlatform (platform) nil (into-array devices)))

(defmacro with-context
  "Defines the context within the inner scope."
  [^CLContext context & body]
  `(let [context# ~context] 
     (with-platform (.getPlatform ^CLContext context#)
       (binding [*context* context#]
	 (enqueue-and-wait
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
	   (.release ^CLContext context#))))))

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

(defn compile-program
  ([source]
     (compile-program (devices) source))
  ([devices source]
     (let [program (.createProgram (context) (into-array devices) (into-array [source]))
	   kernels (.createKernels ^CLProgram program)]
       (zipmap
	 (map #(keyword (.replace (.getFunctionName ^CLKernel %) \_ \-)) kernels)
	 kernels))))

(defmacro with-program [program & body]
  `(binding [*program* ~program]
     ~@body))

(defn local [size]
  (CLKernel$LocalSize. size))

(defn- get-cl-object [x]
  (if (and (instance? clojure.lang.IMeta x) (:cl-object (meta x)))
    (:cl-object (meta x))
    x))

(defn- to-dim-array [x]
  (cond
    (number? x) (int-array [x])
    (cartesian? x) (map #(x %) (range (dimension x)))
    (sequential? x) (into-array x)))

(defn enqueue-kernel
  ([kernel global-size block-size & args]
     (let [kernel ^CLKernel ((program) kernel)]
       (doseq [[idx arg] (indexed (map get-cl-object args))]
	 (.setArg kernel idx arg))
       (.enqueueNDRange kernel (queue) (to-dim-array global-size) *workgroup-size* (make-array CLEvent 0)))))
