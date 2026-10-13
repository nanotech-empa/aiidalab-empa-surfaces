import re


def string_range_to_list(value, shift=-1):
    """Parse atom-index text supporting ranges and common separators.

    Accepts whitespace, commas, semicolons, and spaces around ``..`` ranges.
    With the default ``shift=-1``, user-facing 1-based atom indices are
    converted to Python 0-based indices.
    """
    if value is None:
        return [], False

    normalized = re.sub(r"\s*\.\.\s*", "..", str(value).strip())
    normalized = re.sub(r"[,;]+", " ", normalized)
    if not normalized:
        return [], True

    indices = []
    for item in normalized.split():
        if not re.fullmatch(r"[+-]?\d+(?:\.\.[+-]?\d+)?", item):
            return [], False

        if ".." in item:
            start, end = [int(part) for part in item.split("..")]
            if start > end:
                return [], False
            indices.extend(index + shift for index in range(start, end + 1))
        else:
            indices.append(int(item) + shift)

    return indices, True
