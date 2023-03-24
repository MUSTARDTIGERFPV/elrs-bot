import csv
import sys

def main():
    scores = {}
    with open("votes.csv") as f:
        reader = csv.reader(f, escapechar='\\', quotechar='"')
        for row in reader:
            if len(row) > 2:
                print(f'row {row} failed to parse: too many columns {len(row)} > 2')
                exit()
            message = row[1]
            vote = int(row[0])
            # Init our dict
            if message not in scores:
                scores[message] = 0
            # Tabulate scores
            if vote == 0:
                scores[message] = scores[message] - 1
            elif vote == 1:
                scores[message] = scores[message] + 1
    writer = csv.writer(sys.stdout, delimiter='\t', escapechar='\\', quotechar='"')
    for k, v in scores.items():
        if v > 0:
            writer.writerow([1, k])
        elif v < 0:
            writer.writerow([0, k])

if __name__ == '__main__':
    main()
