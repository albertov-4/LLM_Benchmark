(define (domain citycar)
  (:requirements :strips :typing)
  (:types
    car
    junction
  )

  (:predicates
    (at ?c - car ?j - junction)
    (road ?from - junction ?to - junction)
  )

  (:action move
    :parameters (?c - car ?from - junction ?to - junction)
    :precondition (and
      (at ?c ?from)
      (road ?from ?to)
    )
    :effect (and
      (not (at ?c ?from))
      (at ?c ?to)
    )
  )
)
