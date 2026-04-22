(define (problem tetris-hard-01)
  (:domain tetris)

  (:objects
    piece1 piece2 piece3 - piece
    c1 c2 c3 c4 c5 c6 c7 - cell
  )

  (:init
    (at piece1 c1)
    (at piece2 c3)
    (at piece3 c5)
    (free c2)
    (free c4)
    (free c6)
    (free c7)
    (adjacent c1 c2)
    (adjacent c2 c1)
    (adjacent c2 c3)
    (adjacent c3 c2)
    (adjacent c3 c4)
    (adjacent c4 c3)
    (adjacent c4 c5)
    (adjacent c5 c4)
    (adjacent c5 c6)
    (adjacent c6 c5)
    (adjacent c6 c7)
    (adjacent c7 c6)
  )

  (:goal
    (and
      (at piece1 c4)
      (at piece2 c6)
      (at piece3 c7)
    )
  )
)
