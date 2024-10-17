(define (domain shop)
    (:requirements :typing)

    (:types 
        movable_object region openable - object
    )

    (:constants
        on left_of right_of front_of behind_of - direction 
    )

    (:predicates
        (RobotAt ?region) ; True if robot is at region 
        (RobotHolding ?movable_object) ; True if robot is holding movable_object
        (HandAvailable ) ; True if robot hand is available
        (AtPosition ?subject ?direction ?reference) ; True if subject is at direction of reference
        (IsClosed ?door) ; True if the door is closed 
        (PickOccludedBy ?subject ?occluder) ; True is action (pick, subject) is occluded by occluder 
        (PlaceOccludedBy ?subject ?direction ?reference ?occluder) ; True if action (place, subject, direction, reference) is occluded by occluder
    )

    (:action pick ; example ('pick', 'bottle')
        :parameters (?subject )
        :precondition (and
            (HandAvailable )
            (not (UnsafePick ?subject))
        )

        :effect (and
            (not (HandAvailable ))
            (RobotHolding ?subject)
            (not (AtPosition ?subject ?direction ?reference))
        )
    )

    (:action place ; example ('place', 'bottle', 'behind_of', 'can')
        :parameters (?subject ?direction ?reference)
        :precondition (and
            (RobotHolding ?subject)
            (not (UnsafePlace ?subject ?direction ?reference))
        )

        :effect (and 
            (not (RobotHolding ))
            (HandAvailable )
            (AtPosition ?subject ?direction ?reference)
        )
    )

    (:action open ; example ('open', 'door')
        :parameters (?subject )
        :precondition (and
            (Closed ?subject)
            (HandAvailable )
        )
        :effect (and
            (not (Closed ?subject))
        )
    )

    (:derived (UnsafePick ?subject)
        (exists (?occluder) ((PickOccludedBy ?subject ?occluder)))
    )

    (:derived (UnsafePlace ?subject ?direction ?reference)
        (exists (?occluder) ((PlaceOccludedBy ?subject ?direction ?reference ?occluder)))
    )    
)