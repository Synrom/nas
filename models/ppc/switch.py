# Switch maps from [startnode][endnode][opeartion] -> true/false
Switch = list[list[list[bool]]]


def init_switch(steps: int, operations: int, default: bool) -> Switch:
  switch: Switch = []
  for i in range(steps + 2):
    if i > 1:
      row: list[list[bool]] = []
      for j in range(i):
        ops: list[bool] = [default] * operations
        row.append(ops)
      switch.append(row)
    else:
      switch.append([])
  return switch


def alpha_idx_to_switch_idx(row: list[bool], alpha_idx: int) -> int:
  return [i for i, o in enumerate(row) if o is True][alpha_idx]


def switch_idx_to_alpha_idx(row: list[bool], switch_idx: int) -> int:
  assert row[switch_idx]
  return sum(row[:switch_idx])
