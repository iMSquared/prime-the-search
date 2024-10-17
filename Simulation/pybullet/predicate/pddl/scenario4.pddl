
(:objects
    plate - movable_object
    bottle1 - movable_object
    bottle2 - movable_object
    bottle3 - movable_object
    salter - movable_object
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
    (IsOnRegion salter counter2)
    (IsOnRegion bottle2 counter2)
    (IsOnRegion bottle3 counter2)
    (IsOnRegion bottle1 table1)
    (IsOnRegion plate table1)
    (IsOnRegion tray table2)
    (IsBehindOfObject salter bottle2)
    (IsBehindOfObject salter bottle3)
    (IsBehindOfObject plate bottle1)
    (IsPickOccluded salter bottle2)
    (IsPickOccluded salter bottle3)
    (IsPickOccluded salter kitchen_door)
    (IsPickOccluded bottle2 kitchen_door)
    (IsPickOccluded bottle3 bottle2)
    (IsPickOccluded bottle3 kitchen_door)
    (IsPlaceOccluded salter right_of bottle1 bottle1)
    (IsPlaceOccluded salter right_of bottle1 kitchen_door)
    (IsPlaceOccluded salter right_of bottle1 plate)
    (IsPlaceOccluded salter right_of bottle1 bottle3)
    (IsPlaceOccluded bottle2 on table1 kitchen_door)
    (IsPlaceOccluded bottle2 on table2 kitchen_door)
    (IsPlaceOccluded bottle3 on sink_counter_left bottle2)
    (IsPlaceOccluded bottle3 on minifridge bottle2)
    (IsPlaceOccluded bottle3 on counter1 bottle2)
    (IsPlaceOccluded bottle3 on counter2 bottle2)
    (IsPlaceOccluded bottle3 on shelf_lower bottle2)
    (IsPlaceOccluded bottle3 on table1 kitchen_door)
    (IsPlaceOccluded bottle3 on table1 bottle2)
    (IsPlaceOccluded bottle3 on table2 kitchen_door)
    (IsPlaceOccluded bottle3 on table2 tray)
    (IsPlaceOccluded bottle3 on table2 bottle2)
    (IsPlaceOccluded bottle1 on sink_counter_left kitchen_door)
    (IsPlaceOccluded bottle1 on minifridge kitchen_door)
    (IsPlaceOccluded bottle1 on counter1 kitchen_door)
    (IsPlaceOccluded bottle1 on counter2 kitchen_door)
    (IsPlaceOccluded bottle1 on shelf_lower kitchen_door)
    (IsPlaceOccluded plate on counter1 bottle1)
    (IsPlaceOccluded plate on counter1 kitchen_door)
    (HasEmptyHand )
    (IsClosed kitchen_door)
)

(:goal 
    (and
        ('AtPosition', 'plate', 'on', 'counter1')
        ('AtPosition', 'salter', 'on', 'table2')
        ('AtPosition', 'salter', 'right_of', 'bottle1')
    )
)
