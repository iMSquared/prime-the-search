(:objects
    salter1 - movable_object
    salter2 - movable_object
    salter3 - movable_object
    bottle1 - movable_object
    bottle2 - movable_object
    bottle3 - movable_object
    counter1 - region
    counter2 - region
    shelf_lower - region
    minifridge - region
    sink - region
    table1 - region
    table2 - region
    kitchen_door - openable
)

(:init
    (IsOnRegion bottle2 minifridge)
    (IsOnRegion salter1 counter1)
    (IsOnRegion salter2 counter2)
    (IsOnRegion salter3 table1)
    (IsOnRegion bottle1 table2)
    (IsOnRegion plate1 table2)
    (IsLeftOfObject bottle1 plate1)
    (IsRightOfObject plate1 bottle1)
    (IsPickOccluded salter1 kitchen_door)
    (IsPickOccluded salter2 kitchen_door)
    (IsPickOccluded bottle2 kitchen_door)
    (HasEmptyHand )
    (IsClosed kitchen_door)
)

(:goal 
    (and
        ('AtPosition', 'salter1', 'on', 'counter2')
        ('AtPosition', 'salter2', 'on', 'shelf_lower')
        ('AtPosition', 'bottle3', 'on', 'table1')
        ('AtPosition', 'salter3', 'on', 'table2')
        ('AtPosition', 'bottle1', 'on', 'table1')
        ('AtPosition', 'bottle2', 'on', 'counter2')
    )
)
