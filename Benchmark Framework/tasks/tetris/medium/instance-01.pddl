(define (problem tetris-medium-01)
  (:domain tetris)

  (:objects
    piece1 piece2 - piece
    c1 c2 c3 c4 c5 - cell
  )

  (:init
    (at piece1 c1)
    (at piece2 c3)
    (free c2)
    (free c4)
    (free c5)
    (adjacent c1 c2)
    (adjacent c2 c1)
    (adjacent c2 c3)
    (adjacent c3 c2)
    (adjacent c3 c4)
    (adjacent c4 c3)
    (adjacent c4 c5)
    (adjacent c5 c4)
  )

  (:goal
    (and
      (at piece1 c3)
      (at piece2 c5)
    )
  )
)
