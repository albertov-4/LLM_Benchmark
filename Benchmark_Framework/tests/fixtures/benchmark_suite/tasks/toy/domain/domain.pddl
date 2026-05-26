(define (domain toy)
  (:requirements :strips)
  (:predicates
    (connected)
    (goal-reached)
  )
  (:action move
    :parameters ()
    :precondition (connected)
    :effect (goal-reached)
  )
)
