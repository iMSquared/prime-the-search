(define (domain shop)
    (:requirements :typing)

    (:types 
        movable_object region openable -object
    )

    (:constants
        on left_of right_of front_of behind_of -direction 
    )

    (:predicates
        (RobotAt ?region)
        (RobotHolding ?movable_object)
        (HandAvailable )
        (AtPosition ?subject ?direction ?reference)
        (IsClosed ?door)  
        (PickOccludedBy ?subject ?occluder)
        (PlaceOccludedBy ?subject ?direction ?reference ?occluder)
    )

    (:action pick
        :parameters (?subject )
        :precondition (and
            (HandAvailable )
        )

        :effect (and
            (not (HandAvailable ))
            (RobotHolding ?subject)
        )
    )

    (:action place
        :parameters (?subject ?direction ?reference)
        :precondition (and
            (RobotHolding ?obj)
        )

        :effect (and 
            (not (RobotHolding ))
            (HandAvailable )
            (AtPosition ?subject ?direction ?reference)
        )
    )

    (:action open
        :parameters (?subject )
        :precondition (and
            (IsClosed ?subject)
            (HandAvailable )
        )
        :effect (and
            (not (IsClosed ?subject))
        )
    )
)