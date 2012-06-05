(ns calx.test.lg-env-passing
(:use [calx])
(:use
    ;[gloss.core]
    [clojure.test]))

;;These tests show how an openCL env can be set up and then used for various 
;;computations.

;;Setup openCL env
(def my_devices (available-devices (platform)))
(def my_context (apply create-context (available-devices (platform))))
(def my_queue (lg_create-queue (first my_devices) my_context ))

(def my_openclprog "
__kernel void testaddedkernel(
    __global float *x,
    __global float *y
    )
{
    int gid = get_global_id(0);
    y[gid] = x[gid] - 1.0;
}
__kernel void testaddedkernel2(
    __global float *x,
    __global float *y
    )
{
    int gid = get_global_id(0);
    y[gid] = x[gid] - 1.0;
}
")

(def my_compiled_program 
  (lg_compile-program my_devices my_openclprog my_context))

(def my_openCL_buf1  (lg_create-buffer my_context 10000 :float32-le))
(def my_openCL_buf2  (lg_wrap my_context [1.0 2.0 3.3] :float32-le))
(def my_openCL_buf2a (lg_wrap my_context [5.0 5.2 5.3] :float32-le))
(def my_openCL_buf3  (lg_create-buffer my_context 10000 :float32-le))
(def my_openCL_buf4  (lg_create-buffer my_context 10 :int32-le))

(deftest lg_env_classes
;;Test for classes of non inbuilt clojure objects
(is (= (class my_queue) com.nativelibs4java.opencl.CLQueue))
(is (= (class (:testaddedkernel2 my_compiled_program)) 
       com.nativelibs4java.opencl.CLKernel))
(is (= (class (lg_enqueue-kernel 
                my_queue 
                my_compiled_program 
                :testaddedkernel 
                2 
                my_openCL_buf1 
                my_openCL_buf2))
       com.nativelibs4java.opencl.CLEvent))
(is (= (class 
          (nth @(lg_enqueue-read my_openCL_buf2 my_queue ) 1)) 
       java.lang.Float)))

(time (lg_finish my_queue))


(deftest lg_env_enqueread
;Test of reads from openCL buffers
 (let [my_openCL_buf_testread1  (lg_wrap my_context [-1.0 -1.0 3.3] :float32-le)
       my_openCL_buf_testread2  (lg_wrap my_context [-1 -1 3] :int32-le)]
  (is (= @(lg_enqueue-read my_openCL_buf_testread1 my_queue) 
         (map float [-1.0 -1.0 3.3]))) 
;; cloj1.4 types default to Double
  (is (= @(lg_enqueue-read my_openCL_buf_testread2 my_queue) [-1 -1 3]))
 )
)



(deftest lg_env_enqueue-kernel
  (let [my_openCL_buf_enqueue1 (lg_wrap my_context [-1.0 -1.0 3.3] :float32-le)
        my_openCL_buf_enqueue2 (lg_wrap my_context [-1.1 -1.2 1.3] :float32-le)]
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue1 my_queue)
           (map float [-1.0 -1.0 3.3])))
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue2 my_queue)
           (map float [-1.1 -1.2 1.3])))
    ;;Now we will execure the kernel, global size set to 2.
    (lg_enqueue-kernel my_queue  my_compiled_program
      :testaddedkernel 2 my_openCL_buf_enqueue1 my_openCL_buf_enqueue2)
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue1 my_queue)
           (map float [-1.0 -1.0 3.3])))
    (lg_finish my_queue)  ;; This ensures the enqueue-kernel completes before we read out.
    ;;We see the side effect of the enqueue-kernel
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue2 my_queue)
           (map float [-2.0 -2.0 1.3])))
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue1 my_queue [1 3])
           (map float [-1.0 3.3])))
  )
)

(deftest lg_env_enqueue-overwrite
  (let [my_openCL_buf_enqueue1  (lg_wrap my_context (float-array [-1.0 -1.1 3.3]) :float32-le)
        data_to_overwrite_with (to-buffer [2.9 2.7] :float32-le)]
    (is (= @(lg_enqueue-read   my_openCL_buf_enqueue1 my_queue) (map float [-1.0 -1.1 3.3])))
    (lg_enqueue-overwrite my_openCL_buf_enqueue1 [-2 0] data_to_overwrite_with my_queue)
    (lg_enqueue-barrier my_queue)  ;; This ensures the overwrite completes before we read out.
    ;;We see the side effect of the enqueue-kernel
    (is (= @(lg_enqueue-read my_openCL_buf_enqueue1 my_queue) (map float [2.9 2.7 3.3])))
   )
)

