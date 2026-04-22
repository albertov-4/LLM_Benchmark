(define (problem citycar-hard-01)
  (:domain citycar)

  (:objects
    car1 - car
    j1 j2 j3 j4 j5 j6 j7 j8 - junction
  )

  (:init
    (at car1 j1)
    (road j1 j2)
    (road j2 j3)
    (road j3 j5)
    (road j5 j6)
    (road j6 j8)
    (road j2 j4)
    (road j4 j2)
    (road j3 j1)
    (road j5 j4)
    (road j6 j7)
    (road j7 j6)
  )

  (:goal
    (and
      (at car1 j8)
    )
  )
)
