(define-param dw 4096)
(define-param dh 2048)
(set! geometry-lattice (make lattice (size dw dh no-size)))
(set! geometry (list
                (make block (center 256 0) (size 512 infinity infinity)
                      (material (make dielectric (epsilon 9))))))
(set! sources (list
               (make source
                 (src (make continuous-src (frequency 0.15)))
                 (component Ez)
                 (center 128 0))))
(set! pml-layers (list (make pml (thickness 1.0))))
(set! resolution 1)
(run-until 5000)

