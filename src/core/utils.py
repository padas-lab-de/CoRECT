def count_lines(filepath):
    with open(filepath, "r") as f:
        return sum(1 for _ in f)
