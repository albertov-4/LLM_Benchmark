(define (domain tetris)
  (:requirements :strips :typing)
  (:types
    piece
    cell
  )

  (:predicates
    (at ?p - piece ?c - cell)
    (adjacent ?from - cell ?to - cell)
    (free ?c - cell)
  )

  (:action slide
    :parameters (?p - piece ?from - cell ?to - cell)
    :precondition (and
      (at ?p ?from)
      (adjacent ?from ?to)
      (free ?to)
    )
    :effect (and
      (not (at ?p ?from))
      (at ?p ?to)
      (free ?from)
      (not (free ?to))
    )
  )
)
