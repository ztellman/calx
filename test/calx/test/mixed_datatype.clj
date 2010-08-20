;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns calx.test.mixed-datatype
  (:use [calx] :reload-all)
  (:use [clojure.test]))

(def source
  "struct __attribute__ ((packed)) mixed {
     int val;
     char step;
   };

   __kernel void invert (
       __global const struct mixed *a,
       __global struct mixed *b) {
    int gid = get_global_id(0);
    struct mixed src = a[gid];
    struct mixed m;
    m.val = src.val + src.step;
    m.step = src.step;
    b[gid] = m;
  }")

(deftest invert
  (let [value (with-cl
		(with-program (compile-program source)
		  (let [a (wrap [10 1 0 5] [:int :byte])
			b (mimic a)]
		    (enqueue-read
		      (loop [i 0, a a, b b]
			(if (< 1000 i)
			  a
			  (do
			    (enqueue-kernel :invert 2 a b)
			    (recur (inc i) b a))))))))]
    (is (= @value [[1011 1] [5005 5]]))))
