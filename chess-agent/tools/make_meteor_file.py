#coding: UTF-8
import os
import codecs
import argparse
parser = argparse.ArgumentParser(
    description="...")
parser.add_argument(
    "-i",
    nargs='+',
    dest="inp_pfs",
    type=str)
parser.add_argument(
    "-o",
    dest="out_pf",
    type=str)
args = parser.parse_args()

global pure_references, references
pure_references={}
references = {}
def transit(pf):
    global pure_references, references
    f = codecs.open(pf, 'r', 'UTF-8')
    for i, line in enumerate(f.readlines()):
        if line != '\n':
            if i not in pure_references:
                pure_references[i] = []
                references[i] = []
            references[i].append(line.strip() + ' (id' + str(i) + ')\n')
            pure_references[i].append(line)

for pf in args.inp_pfs:
    transit(pf)

f = codecs.open(args.out_pf, 'w', 'UTF-8')
for ref in pure_references:
    empty_lines = len(args.inp_pfs) - len(pure_references[ref])  # calculate how many empty lines to add (8 max references)
    f.write(''.join(pure_references[ref]))
    if empty_lines > 0:
        f.write('\n' * empty_lines)
f.close()
