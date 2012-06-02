;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns 
  ^{:skip-wiki true}
  calx.core
  (:use
  ;  [clojure.contrib.def :only (defvar defvar-)]
  ;  [clojure.contrib.seq :only (indexed)]
	[clojure.contrib.seq-utils :only [indexed]])
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

;(defvar *platform* nil "The current platform.")
;(defvar *context* nil "The current context.")
;(defvar *queue* nil "The current queue.")
;(defvar *program* nil "The current program")
;(defvar *workgroup-size* nil "The size of the workgroup")
;(defvar *params* nil "The params given to a kernel")
;(defvar *program-template* nil "A function which will return a program, given the current params.")

(def ^:dynamic *platform* "The current platform." nil )
(def ^:dynamic *context* "The current context." nil )
(def ^:dynamic *queue* "The current queue." nil )
(def ^:dynamic *program* "The current program" nil )
(def ^:dynamic *workgroup-size* "The size of the workgroup" nil )
(def ^:dynamic *params* "The params given to a kernel" nil )
(def ^:dynamic *program-template* "A function which will return a program, given the current params." nil )


(defn platform
  "Returns the current platform, or throws an exception if it's not defined."
  []
  (or *platform* (first (available-platforms))))

(defn context
  "Returns the current context, or throws an exception if it's not defined."
  []
  (when-not *context*
    (throw (Exception. "OpenCL context not defined within this scope.")))
  (:context *context*))

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

(defn cache
  "Returns the current data cache"
  []
  (when-not *context*
    (throw (Exception. "OpenCL context not defined within this scope.")))
  (:cache *context*))

(defn program
  "Returns the current program, or throws an exception if it's not defined."
  []
  (cond
    *program-template* (*program-template*)
    *program* *program*
    :else  (throw (Exception. "OpenCL program not defined within this scope."))))

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

(defn create-context
  "Creates a context which uses the specified devices.  Using more than one device is not recommended."
  [& devices]
  {:context (.createContext ^CLPlatform (platform) nil (into-array devices))
   :cache (ref [])})

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

(defn enqueue-wait-for
  "Enqueues a barrier which will halt execution until all events given as parameters have completed."
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

;;;

(defn- eval-templates
  "Evaluates anything inside <<<...>>> as Clojure code."
  [s]
  (let [parts (.split (str s " ") "<<<|>>>")]
    (when (even? (count parts))
      (throw (Exception. "Mismatched <<< and >>> delimiters.")))
    (let [parts (partition-all 2 parts)
	  literals (map first parts)
	  code (filter #(and % (not (empty? %))) (map second parts))
	  code (map #(eval (read-string %)) code)
	  combined (interleave (cons "" code) literals)]
      (.trim ^String (apply str combined)))))

(defn compile-program
  "Compiles a OpenCL program, which contains 1 or more kernels."
  ([source]
     (compile-program (devices) source))
  ([devices source]
     (let [program (.createProgram (context) (into-array devices) (into-array [(eval-templates source)]))
	   kernels (.createKernels ^CLProgram program)]
       (zipmap
	 (map #(keyword (.replace (.getFunctionName ^CLKernel %) \_ \-)) kernels)
	 kernels))))

(defn local [size]
  (CLKernel$LocalSize. size))

(defn- get-cl-object [x]
  (if (and (instance? clojure.lang.IMeta x) (:cl-object (meta x)))
    (:cl-object (meta x))
    x))

(defn- to-dim-array [x]
  (cond
    (number? x) (int-array [x])
    (sequential? x) (into-array x)))

(defn enqueue-kernel
  ([kernel global-size & args]
     (let [kernel ^CLKernel ((program) kernel)]
       (doseq [[idx arg] (indexed (map get-cl-object args))]
	 (.setArg kernel idx arg))
       (.enqueueNDRange kernel (queue) (to-dim-array global-size) *workgroup-size* (make-array CLEvent 0)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; This section is an alternative implementation designed to accept as parameter
; the openCL environment under which execution is to occour.

(defn lg_enqueue-barrier
  "Ensures that all previously enqueued commands will complete before new commands 
   will begin, on a specific queue"
  [queue]
  (.enqueueBarrier ^CLQueue queue))

(defn lg_create-queue
  "Creates a queue, against a specific device and context"
  ([]
     (create-queue (first (devices)) context ))
  ([^CLDevice device context & properties]
     (.createQueue
       device
       (:context context)   ;;(context)
       (if (empty? properties)
	 (make-array CLDevice$QueueProperties 0)
	 (into-array properties)))))


(defn lg_finish
  "Halt execution until all enqueued operations are complete on a specific queue"
  ([q]
     (.finish ^CLQueue q)))

(defn lg_compile-program
  "Compiles a OpenCL program, which contains 1 or more kernels, compiles under
   a specific context"
  [devices source context]
      (let [program (.createProgram (:context context) (into-array devices) (into-array [(eval-templates source)]))
	          kernels (.createKernels ^CLProgram program)]
       (zipmap
	 (map #(keyword (.replace (.getFunctionName ^CLKernel %) \_ \-)) kernels)
	 kernels)))

(defn lg_enqueue-kernel
   "enqueue a kernel on a specific queue and from a specific compiled program"
  ([queue program kernel global-size & args]
     (let [kernel ^CLKernel (program kernel)]
       (doseq [[idx arg] (indexed (map get-cl-object args))]
	 (.setArg kernel idx arg))
       (.enqueueNDRange kernel queue (to-dim-array global-size) *workgroup-size* (make-array CLEvent 0)))))

