import os
import csv


# filename='results.csv'
def setup_csv(filename):
    # Check if file exists
    if os.path.exists(filename):
        # Clear the file
        open(filename, 'w').close()

    # Open the file in append mode
    file = open(filename, 'a', newline='')
    writer = csv.writer(file)

    # Write headers
    writer.writerow(['Episode','Reward','Path','Time Taken'])

    return file, writer
# file, writer = setup_csv(filename)