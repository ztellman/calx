;;   Copyright (c) Zachary Tellman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;;   which can be found in the file epl-v10.html at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns calx.byte-buffer
  (:use [calx.data] [clojure.contrib.combinatorics] :reload-all)
  (:use [clojure.test]))

(def types [:byte :short :int :long :float :double])

(deftest to-and-from
  (doseq [sig (permutations types)]
    (let [data (range (count sig))]
      (is (= data (first (from-buffer (to-buffer data sig) sig)))))))
