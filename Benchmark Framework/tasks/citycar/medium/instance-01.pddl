(define (problem citycar-medium-01)
  (:domain citycar)

  (:objects
    car1 - car
    j1 j2 j3 j4 j5 - junction
  )

  (:init
    (at car1 j1)
    (road j1 j2)
    (road j2 j3)
    (road j3 j5)
    (road j2 j4)
    (road j4 j2)
  )

  (:goal
    (and
      (at car1 j5)
    )
  )
)
