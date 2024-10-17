(:objects
    coke - movable_object
    sprite - movable_object
    mango_juice - movable_object
    dr_pepper - movable_object
    pork_grill - movable_object
    beef_grill - movable_object
    kitchen_door - openable
    counter1 - region
    counter2 - region
    counter3 - region
    counter5 - region
)

(:init
    (HasEmptyHand )
    (AtPosition coke on counter2)
    (AtPosition beef_grill on counter2)
    (AtPosition sprite on counter3)
    (AtPosition pork_grill on counter3)
)

(:goal 
    (and
        (AtPosition coke on counter1)
        (AtPosition sprite on counter1)
        (AtPosition beef_grill on counter1)
    )
)
