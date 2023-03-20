def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    "Process function for splitting all the punctuation"
    out = ''
    for word_ in text.split(" "):
        w=word_; end_chars = ''
        while any(elem in w for elem in punctuation) and len(w) > 1:
            if w[-1] in punctuation: end_chars = w[-1] + end_chars; w = w[:-1]
            else: break
        try:
            word_arpa = thisdict[w.upper()]
            w = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + w + end_chars).strip()
    if EOS_Token and out[-1] != ";": out += ";"
    return out