(define (problem tetris-easy-01)
  (:domain tetris)

  (:objects
    piece1 - piece
    c1 c2 c3 - cell
  )

  (:init
    (at piece1 c1)
    (free c2)
    (free c3)
    (adjacent c1 c2)
    (adjacent c2 c1)
    (adjacent c2 c3)
    (adjacent c3 c2)
  )

  (:goal
    (and
      (at piece1 c3)
    )
  )
)
