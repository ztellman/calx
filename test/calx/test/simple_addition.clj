;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns calx.test.simple-addition
  (:use [calx])
  (:use [clojure.test]))

(def source
  "__kernel void vec_add (
       __global const float *a,
       __global const float *b,
       __global float *c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] + b[gid];
  }")

(deftest simple-addition
  (let [value (with-cl
		(with-program (compile-program source)
		  (let [a (wrap [1.0 2.0 3.0] :float32-le)
			b (wrap [1.0 2.0 3.0] :float32-le)
			c (mimic a)]
		    (enqueue-kernel :vec-add 3 a b c)
		    (enqueue-read c))))]
    (is (= [2.0 4.0 6.0] @value))))
