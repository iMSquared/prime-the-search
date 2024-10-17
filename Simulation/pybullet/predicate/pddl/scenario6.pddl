(:objects
    coke - movable_object
    sprite - movable_object
    beer - movable_object
    beef_grill - movable_object
    kitchen_door - openable
    counter1 - region
    counter2 - region
    shelf_lower - region
    minifridge - region
    sink - region
    table1 - region
    table2 - region
)

(:init
    (HasEmptyHand )
    (AtPosition coke on counter1)
    (AtPosition beef_grill on counter2)
    (AtPosition sprite on table2)
    (AtPosition beer on table2)
    (AtPosition fanta on table2)
    (AtPosition beer right_of sprite)
    (AtPosition beer behind_of fanta)
    (AtPosition sprite behind_of fanta)
    (AtPosition wine on table1)
)

(:goal 
    (and
        (AtPosition coke on counter2)
        (AtPosition wine on counter2)
        (AtPosition beef_grill on counter2)
    )
)
