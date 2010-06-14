(ns calx.core
  (:use [clojure.contrib.def :only (defvar)])
  (:import [com.nativelibs4java.opencl
	    JavaCL CLContext CLPlatform
	    CLDevice CLDevice$QueueProperties
	    CLQueue CLEvent]))

;;;

(defvar *platform* nil)
(defvar *context* nil)
(defvar *devices* nil)
(defvar *queue* nil)

(defn platform []
  (when-not *platform*
    (throw (Exception. "OpenCL platform not defined within this scope.")))
  *platform*)

(defn context []
  (when-not *context*
    (throw (Exception. "OpenCL context not defined within this scope.")))
  *context*)

(defn devices []
  (when-not *devices*
    (throw (Exception. "OpenCL devices not defined within this scope.")))
  *devices*)

(defn queue []
  (when-not *queue*
    (throw (Exception. "OpenCL queue not defined within this scope.")))
  *queue*)

;;;

(defn get-platforms []
  (seq (JavaCL/listPlatforms)))

(defn get-all-devices
  ([]
     (get-all-devices (platform)))
  ([platform]
     (seq (.listAllDevices #^CLPlatform platform true))))

(defn get-cpu-devices
  ([]
     (get-cpu-devices (platform)))
  ([platform]
     (seq (.listCPUDevices #^CLPlatform platform true))))

(defn get-gpu-devices
  ([]
     (get-gpu-devices (platform)))
  ([platform]
     (seq (.listGPUDevices #^CLPlatform platform true))))

(defn get-best-device
  ([]
     (get-best-device (platform)))
  ([platform]
     (.getBestDevice #^CLPlatform platform)))

(defn get-version
  ([]
     (get-version (platform)))
  ([platform]
     (.getVersion #^CLPlatform platform)))

;;;

(defprotocol HasEvent
  (event [e]))

(extend-type CLEvent
  HasEvent
  (event [e] e))

(defn wait-for [& events]
  (CLEvent/waitFor (into-array (map event events))))

(defn create-queue
  ([]
     (create-queue (first (devices))))
  ([#^CLDevice device & properties]
     (.createQueue
      device
      (context)
      (if (empty? properties)
	(make-array CLDevice$QueueProperties 0)
	(into-array properties)))))

(defmacro with-queue [& body]
  `(binding [*queue* (create-queue)]
     ~@body))

(defn finish
  "Wait for all enqueued operations to complete."
  []
  (.finish #^CLQueue (queue)))

(defn enqueue-barrier
  []
  (.enqueueBarrier #^CLQueue (queue)))

(defn enqueue-marker
  []
  (.enqueueMarker #^CLQueue (queue)))

(defn enqueue-wait
  [& events]
  (.enqueueWaitForEvents #^CLQueue (queue) (into-array (map event events))))

;;;

(defmacro with-platform [platform & body]
  `(binding [*platform* (or ~platform *platform* (first (get-platforms)))]
     ~@body))

(defn create-context [& devices]
  (with-platform (or *platform* (first (get-platforms)))
    (.createContext #^CLPlatform (platform) nil (into-array devices))))

(defmacro with-context [#^CLContext context & body]
  `(with-platform nil
     (let [context# ~context]
       (with-platform (.getPlatform #^CLContext context#)
	 (binding [*devices* (seq (.getDevices #^CLContext context#))
		   *context* context#]
	   (with-queue
	     ~@body))))))

(defmacro with-devices [devices & body]
  `(with-platform nil
     (let [context# (apply create-context ~devices)]
       (try
	 (with-context context#
	   ~@body)
	 (finally
	  (.release #^CLContext context#))))))

(defmacro with-cpu [& body]
  `(with-devices [(first (get-cpu-devices))] ~@body))

(defmacro with-gpu [& body]
  `(with-devices [(first (get-gpu-devices))] ~@body))

(defmacro with-cl [& body]
  `(with-devices [(get-best-device)] ~@body))

;;;
