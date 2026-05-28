"""
Converts a set of predicted notebook (logs generated from data analysis model evalution) 
to a corresponding set of IPython notebooks.
"""

import argparse
import glob
import nbformat
import re
from pathlib import Path

def expand_filelist(pattern_list: list[str]) -> list[str]:
    """
    Expands a list of file patterns to a flat master list of files.
    """
    file_list = []
    for pattern in pattern_list:
        prev_len = len(file_list)
        file_list.extend(glob.glob(pattern))
        if len(file_list) == prev_len:
            raise FileNotFoundError(f"No files found for pattern (check your paths?): {pattern}")
    return file_list

def log_to_nbformat(log_file: str) -> nbformat.NotebookNode:
    """
    Reads a log file, extracts the code cells, and returns an IPython notebook.
    """
    nb = nbformat.v4.new_notebook()
    with open(log_file, 'r') as f_in:
        contents = f_in.read()

        # Strip header
        contents = re.split(r'^---\n', contents, flags=re.MULTILINE, maxsplit=1)
        header = contents[0].strip()
        contents = contents[1].strip()
        nb.cells = [nbformat.v4.new_markdown_cell(source=header)]
        # Strip each code cell's output footer
        contents = re.split(r'^## Output .*?---\n', contents, flags=re.DOTALL | re.MULTILINE)
        contents[-1] = re.sub(r'## Output .*\Z', '', contents[-1], flags=re.DOTALL)

        nb.cells.extend([nbformat.v4.new_code_cell(source=cell) for cell in contents])
    
    return nb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_files", nargs='+',
        help="The log files to convert to an IPython notebook. Patterns allowed.")
    args = parser.parse_args()

    file_list = expand_filelist(args.log_files)

    # Convert log for each file
    for file in file_list:
        print(f"Converting {file} to IPython notebook...")
        nb = log_to_nbformat(file)
        nb_path = Path(file).with_suffix('.ipynb')
        nbformat.write(nb, nb_path)