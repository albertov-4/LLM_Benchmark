(define (problem citycar-easy-01)
  (:domain citycar)

  (:objects
    car1 - car
    j1 j2 j3 - junction
  )

  (:init
    (at car1 j1)
    (road j1 j2)
    (road j2 j3)
  )

  (:goal
    (and
      (at car1 j3)
    )
  )
)
