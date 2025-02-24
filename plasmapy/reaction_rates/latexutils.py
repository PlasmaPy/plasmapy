import math
import pandas as pd

table_placeholder = "–"


def JFE_comply(latex):
    """JFE uses the old style horizontal rule, see:
    https://tex.stackexchange.com/questions/156122/booktabs-what-is-the-difference-between-toprule-and-hline
    so we have to replace the more modern horizontal rules with the older ones
    """
    latex = latex.replace("\\toprule", "\\hline\\noalign{\\smallskip}")
    latex = latex.replace(
        "\\midrule", "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}"
    )
    latex = latex.replace("\\bottomrule", "\\noalign{\\smallskip}\\hline")
    return latex


def full_width_table(latex):
    """Hack to make tables full page width"""
    latex = latex.replace("\\begin{table}", "\\begin{table*}[p]")
    latex = latex.replace("\\end{table}", "\\end{table*}")
    return latex


def sideways_table(latex):
    """Hack to make tables sideways"""
    latex = latex.replace(r"\begin{table*}[p]", r"\begin{sidewaystable*}[p]")
    latex = latex.replace(r"\end{table*}", r"\end{sidewaystable*}")

    # Fix column spacing so columns are evenly spaced across the page
    latex = latex.replace(
        r"\begin{tabular}{", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
    )
    latex = latex.replace(r"\end{tabular}", r"\end{tabular*}")

    return latex


def include_table_footnote(latex, footnote):
    r"""Hack to include custom table footnote after "\end{tabular}" """
    # Regular tables
    latex = latex.replace(
        r"\end{tabular}",
        r"""\end{tabular}
                              \raggedright
                              \footnotesize{%s}"""
        % footnote,
    )

    # Full width sideways tables
    latex = latex.replace(
        r"\end{tabular*}",
        r"""\end{tabular*}
                              \raggedright
                              \footnotesize{%s}"""
        % footnote,
    )
    return latex


def display_Q(Q):
    if Q == float("inf"):
        return r"$\infty$"
    else:
        return str(int(Q))


def siunitx_num(s):
    """Format string s to return
    '\num{s}'
    so the siunitx latex package correctly display scientific notation
    """
    if s == table_placeholder or pd.isnull(s):
        return table_placeholder
    else:
        return r"\num{" + str(s) + "}"


def cite(bibtex_strings):
    citation = rf"\onlinecite{{{','.join(bibtex_strings)}}}"
    return citation


def CustomLogarithmicFormatter(x, pos=None):
    """This formatter is intended for non-scientific notation tick labels
    on logarithmic axes. It avoids displaying trailing zeros by adjusting
    the rounding for values less than 1.
    """
    if x < 1:
        round_places = int(-1 * math.log10(x))
        x_formatted = format(x, f"0.{round_places}f")
    else:
        x_formatted = format(x, "0.0f")
    return x_formatted
