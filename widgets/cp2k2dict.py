import numpy as np


def CP2K2DICT(input_lines=""):
    input_dict = {}
    lines = input_lines.splitlines()
    i_am_at = [input_dict]
    for l in lines:
        fc = len(l) - len(l.lstrip())
        if l[fc] not in ["#", "!"]:
            has_arg = len(l.split()) > 1
            key = l.split()[0].lstrip("&")
            if "&" in l and "END" not in l:
                if key in i_am_at[-1].keys():
                    if type(i_am_at[-1][key]) is not type([]):
                        i_am_at[-1][key] = [i_am_at[-1][key]]
                    if has_arg:
                        i_am_at[-1][key].append(
                            {"_": " ".join([str(s) for s in l.split()[1:]])}
                        )
                    else:
                        i_am_at[-1][key].append({})
                    i_am_at.append(i_am_at[-1][key][-1])
                else:
                    if has_arg:
                        i_am_at[-1][key] = {
                            "_": " ".join([str(s) for s in l.split()[1:]])
                        }
                    else:
                        i_am_at[-1][key] = {}
                    i_am_at.append(i_am_at[-1][key])
            elif "&" in l and "END" in l:
                if type(i_am_at[-1]) is not type([]):
                    i_am_at = i_am_at[0:-1]
                else:
                    i_am_at = i_am_at[0 : -len(i_am_at[-1])]
            else:
                while (
                    key in i_am_at[-1].keys()
                ):  # fixes teh case where inside &KINDS we have BASIS .. and BASIS RI_AUX ..
                    key = key + " "
                if has_arg:
                    i_am_at[-1].update({key: " ".join([str(s) for s in l.split()[1:]])})
                else:
                    i_am_at[-1].update({key: ""})
    if len(i_am_at) > 1:
        msg = "there are unclosed cards"
    else:
        msg = ""

    return msg, input_dict
