import os
import pandas as pd
import argparse
import numpy as np

class post_processing:

    def process_single_csv(self, csv_file):
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Create a new column 'Folder' by splitting the 'FileName' column on the underscore and taking the first part
        df['Folder'] = df['FileName'].apply(lambda x: x.rsplit('_', 1)[0])

        # Group the DataFrame by the 'Folder' column
        grouped = df.groupby('Folder')
        processed_df = pd.DataFrame()

        for name, group in grouped:
            group = group.reset_index(drop=True)
            group = self.post_process(group)
            processed_df = pd.concat([processed_df, group], axis = 0).reset_index(drop=True)

        processed_df = processed_df.drop(columns=['Folder'])
        processed_df.to_csv("processed.csv", index=False)

    def process_csv_files(self, root_dir):
        # Iterate over each folder in the root directory
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            # Check if the folder contains a CSV file
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                # Read the CSV file
                df = pd.read_csv(os.path.join(folder_path, csv_file))

                # Post process the DataFrame
                df = self.post_process(df)

                # Save the updated CSV file
                df.to_csv(os.path.join(folder_path, csv_file), index=False)

    def propagate_scalpel(self, data):
        # If 'scalpel' label exists in 'Overall Task' column, make the above 10 rows under 'Overall Task' column all 'scalpel'
        scalpel_indices = data[data['Overall Task'] == 'scalpel'].index
        for index in scalpel_indices:
            if index >= 10:
                data.loc[index - 10:index, 'Overall Task'] = 'scalpel'
        return data

    def propagate_dilator(self, data):
        # If 'dilator' label exists in 'Overall Task' column, make the below 10 rows under 'Overall Task' column all 'dilator'
        dilator_indices = data[data['Overall Task'] == 'dilator'].index
        for index in dilator_indices:
            if index <= len(data) - 11:
                data.loc[index:index + 10, 'Overall Task'] = 'dilator'

        # Change all 'Overall Task' column's later rows after 'dilator' where it has values of 'anesthetic' or 'insert_needle' into 'insert_catheter'
        dilator_indices = data[data['Overall Task'] == 'dilator'].index
        for index in dilator_indices:
            if index <= len(data) - 11:
                data.loc[index + 11:, 'Overall Task'] = data.loc[index + 11:, 'Overall Task'].replace(
                    {'anesthetic': 'insert_catheter', 'insert_needle': 'insert_catheter'})
        return data

    def smooth_sequence(self, data):
        # Reset the index of the DataFrame
        data = data.reset_index(drop=True)

        # Initialize the previous label and sequence length
        prev_label = data.loc[0, 'Overall Task']
        seq_length = 0

        # Iterate over the DataFrame
        for index, row in data.iterrows():
            # If the current label is the same as the previous label, increment the sequence length
            if row['Overall Task'] == prev_label:
                seq_length += 1
            else:
                # If the sequence length is less than or equal to 9, change the label of the sequence to the previous sequence's label
                if seq_length <= 9 and index > 9:
                    data.loc[index - seq_length:index, 'Overall Task'] = data.loc[
                        index - seq_length - 1, 'Overall Task']
                # Update the previous label and reset the sequence length
                prev_label = row['Overall Task']
                seq_length = 1

        # Check the last sequence
        if seq_length <= 9 and seq_length < len(data):
            data.loc[len(data) - seq_length:, 'Overall Task'] = data.loc[len(data) - seq_length - 1, 'Overall Task']

        return data

    def change_first_sequence(self, data):
        # Initialize the counter
        seq_length = 0
        first_label = data.loc[0, 'Overall Task']
        next_seq_start = None

        # Iterate over the DataFrame rows
        for index, row in data.iterrows():
            # If the current row's 'Overall Task' is the same as the first row's 'Overall Task', increment the counter
            if row['Overall Task'] == first_label:
                seq_length += 1
                data.loc[index, 'Overall Task'] = 'Cross-section'
            else:
                next_seq_start = index
                # If the current row's 'Overall Task' is not the same as the first row's 'Overall Task', break the loop
                break

        if seq_length <= 10 and next_seq_start is not None:
            next_seq_label = data.loc[next_seq_start, 'Overall Task']
            next_seq_length = 0

            for index in range(next_seq_start, len(data)):
                if data.loc[index, 'Overall Task'] == next_seq_label:
                    next_seq_length += 1
                else:
                    break

            if next_seq_label != 'Cross-section':
                data.loc[next_seq_start:next_seq_start + next_seq_length, 'Overall Task'] = 'Cross-section'



        return data

    def change_label_according_to_position(self, data):
        # Calculate the indices that represent the 50%, 30%, and 20% positions of the DataFrame
        half_length = len(data) // 2
        thirty_percent_length = len(data) * 3 // 10
        twenty_percent_length = len(data) // 5


        # Change the labels in the 'Overall Task' column as required
        data.loc[:half_length, 'Overall Task'] = data.loc[:half_length, 'Overall Task'].replace(
            {'insert_catheter': 'insert_guidewire'})
        data.loc[half_length:, 'Overall Task'] = data.loc[half_length:, 'Overall Task'].replace(
            {'anesthetic': 'insert_catheter'})

        # Find the last 10 occurrences of 'insert_guidewire' in the 'Overall Task' column
        last_ten_insert_guidewire_indices = data[data['Overall Task'] == 'insert_guidewire'].tail(10).index
        # If there are at least 10 'insert_guidewire', change them to 'remove_guidewire'
        if len(last_ten_insert_guidewire_indices) == 10:
            data.loc[last_ten_insert_guidewire_indices, 'Overall Task'] = 'remove_guidewire'
        #
        # data.loc[-thirty_percent_length:, 'Overall Task'] = data.loc[-thirty_percent_length:, 'Overall Task'].replace(
        #     {'insert_guidewire': 'insert_catheter'})
        # data.loc[-twenty_percent_length:, 'Overall Task'] = data.loc[-twenty_percent_length:, 'Overall Task'].replace(
        #     {'insert_needle': 'insert_catheter'})
        return data

    def change_between_sequence(self, data):
        # Find the last occurrence of 'anesthetic' that has 'insert_needle' after it
        anesthetic_indices = data[data['Overall Task'] == 'anesthetic'].index
        last_anesthetic_index = None
        for index in reversed(anesthetic_indices):
            if 'insert_needle' in data.loc[index:, 'Overall Task'].values:
                last_anesthetic_index = index
                break

        # Find the first occurrence of 'insert_needle' sequence after the last_anesthetic_index
        if last_anesthetic_index is not None:
            insert_needle_indices = data.loc[last_anesthetic_index:, 'Overall Task'][
                data.loc[last_anesthetic_index:, 'Overall Task'] == 'insert_needle'].index
            if not insert_needle_indices.empty:
                first_insert_needle_index = insert_needle_indices[0]
            else:
                first_insert_needle_index = None

        # If both indices are found, change all the labels in the 'Overall Task' column between these two indices to 'cross-section'
        if last_anesthetic_index is not None and first_insert_needle_index is not None:
            data.loc[last_anesthetic_index:first_insert_needle_index, 'Overall Task'] = 'Cross-section'

        return data

    def change_last_sequence(self, data):
        # Change the 'Overall Task' label into 'nothing' for the last 30 rows
        data.loc[data.tail(30).index, 'Overall Task'] = 'nothing'
        return data

    def post_process(self, data):
        data = self.propagate_scalpel(data)
        data = self.propagate_dilator(data)
        data = self.smooth_sequence(data)
        data = self.change_first_sequence(data)
        data = self.change_label_according_to_position(data)
        data = self.change_between_sequence(data)
        data = self.change_last_sequence(data)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, help="Specify the root directory path")
    parser.add_argument("-csv", "--main_csv_file", type=str, help="Specify the main csv path")
    args = parser.parse_args()

    pp = post_processing()

    if args.root_dir:
        pp.process_csv_files(args.root_dir)
    elif args.main_csv_file:
        pp.process_single_csv(args.main_csv_file)