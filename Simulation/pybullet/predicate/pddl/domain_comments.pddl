(define (domain shop)
    (:requirements :typing)

    (:types 
        movable_object region openable - object
    )

    (:constants
        on left_of right_of front_of behind_of - direction 
    )

    (:predicates
        (IsAtDirectionOfObject ?obj1 - movable_object ?dir - direction ?ref - movable_object) ; True if position of obj1 is at direction of ref
        (IsOnRegion ?obj - movable_object ?reg - region)  ; True if position of obj is on reg
        (IsClosed ?door - openable)  ; True if door is closed
        (IsPickOccluded ?obj - movable_object ?occluder - object)  ; True if occluder is causing occlusion when trying to pick obj on reg
        (IsPlaceOccluded ?obj - movable_object ?dir - direction ?ref - object ?occluder - object)  ; True if occluder is causing occlusion when trying to place  holding obj at dir of ref on reg
        (IsHolding ?obj - movable_object) ; True if the agent is holding obj
        (HasEmptyHand )  ; True if the agent has an empty hand
    )

    (:action pick
        :parameters (?obj - movable_object ?reg - region) ; pick obj that is on reg 
        :precondition (and
            (HasEmptyHand ) ; Agent should have an empty hand available for action
            (IsOnRegion ?obj ?reg) ; Given reg should be where ref is on 
        )
        :effect (and
            (IsHolding ?obj) ; Agent is holding obj
            (not (IsOnRegion ?obj ?reg)) ; obj is not on reg 
        )
    )

    (:action place
        :parameters (?obj - movable_object ?dir - direction ?ref - object ?reg - region) ; place holding obj at dir of ref that is on region 
        :precondition (and
            (IsHolding ?obj) ; Agent should be holding obj 
            (IsOnRegion ?ref ?reg) ; Given reg should be where ref is on 
        )
        :effect (and 
            (IsOnRegion ?obj ?reg) ; obj on reg where the ref is on 
        )
    )

    (:action open
        :parameters (?door - openable) ; open door
        :precondition (and
            (HasEmptyHand ) ; Agent should have an empty hand available for action
            (IsClosed ?door) ; door should be closed
        )
        :effect (and 
            (not (IsClosed ?door)) ; door is not closed
        )
    )
)