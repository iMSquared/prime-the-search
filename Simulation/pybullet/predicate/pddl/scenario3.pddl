(:objects
    beer - movable_object
    milk - movable_object
    coke - movable_object
    sprite - movable_object
    fanta - movable_object
    water - movable_object
    wine - movable_object
    orange_juice - movable_object
    grape_juice - movable_object
    whiskey - movable_object
    welchies - movable_object
    cocoa - movable_object
    kitchen_door - openable
    counter1 - region
    table1 - region

)

(:init
    (HasEmptyHand )
    (AtPosition beer on table1)
    (AtPosition milk on table1)
    (AtPosition coke on table1)
    (AtPosition sprite on table1)
    (AtPosition fanta on table1)
    (AtPosition water on table1)
    (AtPosition wine on table1)
    (AtPosition orange_juice on table1)
    (AtPosition grape_juice on table1)
    (AtPosition whiskey on table1)
    (AtPosition welchies on table1)
)

(:goal 
    (and
        (AtPosition milk on counter2)
        (AtPosition coke on counter2)
    )
)
