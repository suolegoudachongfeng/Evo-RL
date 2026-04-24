def parse_buttons(text: str) -> dict[str, bool | tuple[float, ...]]:
    split_text = text.split(",")
    buttons: dict[str, bool | tuple[float, ...]] = {}
    if "R" in split_text:
        split_text.remove("R")
        buttons.update(
            {
                "A": False,
                "B": False,
                "RThU": False,
                "RJ": False,
                "RG": False,
                "RTr": False,
            }
        )

    if "L" in split_text:
        split_text.remove("L")
        buttons.update({"X": False, "Y": False, "LThU": False, "LJ": False, "LG": False, "LTr": False})

    for key in list(buttons.keys()):
        if key in split_text:
            buttons[key] = True
            split_text.remove(key)

    for elem in split_text:
        split_elem = elem.split(" ")
        if len(split_elem) < 2:
            continue
        key = split_elem[0]
        value = tuple(float(x) for x in split_elem[1:])
        buttons[key] = value
    return buttons
