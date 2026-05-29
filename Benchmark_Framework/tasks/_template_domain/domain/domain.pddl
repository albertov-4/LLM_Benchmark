; Template PDDL domain.
; Replace the domain name, requirements, predicates, functions, and actions.

(define (domain template-domain)
  (:requirements :strips :typing)

  (:types
    item
  )

  (:predicates
    (ready ?x - item)
    (done ?x - item)
  )

  (:action complete
    :parameters (?x - item)
    :precondition (ready ?x)
    :effect (and
      (done ?x)
    )
  )
)
