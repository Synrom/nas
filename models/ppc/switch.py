# Switch maps from [startnode][endnode][opeartion] -> true/false
Switch = list[list[list[bool]]]


def init_switch_all_true(steps: int, operations: int) -> Switch:
  return [[[True] * operations] * i if i > 1 else [] for i in range(steps + 2)]


def init_switch_all_false(steps: int, operations: int) -> Switch:
  return [[[False] * operations] * i if i > 1 else [] for i in range(steps + 2)]
