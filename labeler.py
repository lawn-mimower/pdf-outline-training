import pandas as pd
import os
import argparse

class CSVLabeler:
    """
    A command-line tool to efficiently label consolidated text spans in a CSV file.
    """
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = self._load_data()
        self.last_index = -1

        # --- MODIFIED: Updated list of predefined labels as requested ---
        self.PREDEFINED_LABELS = [
            'TITLE', 'H1', 'H2', 'H3', 'BODY'
        ]

    def _load_data(self):
        """Loads data, resuming from an existing output file if possible."""
        try:
            input_df = pd.read_csv(self.input_csv)
        except FileNotFoundError:
            print(f"❌ Error: Input file not found at '{self.input_csv}'")
            exit()

        if os.path.exists(self.output_csv):
            print(f"📖 Resuming from existing file '{self.output_csv}'...")
            return pd.read_csv(self.output_csv)
        else:
            print(f"✨ Starting a new labeling session...")
            input_df['label'] = 'UNLABELED'
            return input_df

    def _save_progress(self):
        """Saves the DataFrame to the output CSV."""
        self.df.to_csv(self.output_csv, index=False)

    def _display_span(self, index, row):
        """Clears the screen and displays the current span and its context."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        total_spans = len(self.df)
        progress = f"📊 Span {index + 1}/{total_spans}"
        print(f"{'='*30}\n{progress}\n{'='*30}")

        print("\n📝 TEXT TO LABEL:")
        print(f"   \"{row['text']}\"")

        print("\n✨ KEY FEATURES:")
        print(f"   - Font Size: {row['font_size']:.2f} | Bold: {'Yes' if row['is_bold'] else 'No'}")
        print(f"   - Page: {row['page']} | Is Centered: {'Yes' if row['is_centered'] else 'No'}")
        
        print("\n🏷️ AVAILABLE LABELS:")
        for i, label in enumerate(self.PREDEFINED_LABELS, 1):
            print(f"   {i}. {label}")

        print("\nCOMMANDS: (u)ndo, (s)kip, (q)uit, or enter a custom label")
        print("-" * 30)

    def run(self):
        """The main labeling loop."""
        i = 0
        # Find first unlabeled span to start/resume
        try:
            start_index = self.df[self.df['label'] == 'UNLABELED'].index[0]
            i = start_index
        except IndexError:
            print("🎉 All spans are already labeled! Nothing to do.")
            return

        while i < len(self.df):
            row = self.df.iloc[i]
            
            self._display_span(i, row)
            
            user_input = input("Enter label number, command, or custom label: ").strip().lower()

            if user_input in ['q', 'quit']:
                print("Quitting and saving progress...")
                self._save_progress()
                break
            
            elif user_input in ['s', 'skip']:
                self.last_index = i
                i += 1
                continue

            elif user_input in ['u', 'undo']:
                if self.last_index != -1:
                    print("⏪ Undoing last label...")
                    self.df.at[self.last_index, 'label'] = 'UNLABELED'
                    i = self.last_index
                else:
                    print("⚠️ Cannot undo. No previous action in this session.")
                    input("Press Enter to continue...")
                continue
            
            elif user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(self.PREDEFINED_LABELS):
                    self.df.at[i, 'label'] = self.PREDEFINED_LABELS[choice - 1]
                    self.last_index = i
                    i += 1
                else:
                    print("⚠️ Invalid number. Please try again.")
                    input("Press Enter to continue...")
            
            else: # Custom label
                if user_input:
                    # Use user's casing for custom labels, e.g., 'Table' or 'TABLE'
                    self.df.at[i, 'label'] = user_input.strip()
                    self.last_index = i
                    i += 1
                else:
                    print("⚠️ Please enter a label or a command.")
                    input("Press Enter to continue...")

            # Save progress every 10 labels
            if i > 0 and i % 10 == 0:
                self._save_progress()
        
        self._save_progress()
        print("\n✅ Labeling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line tool for labeling CSV data.")
    parser.add_argument("input_csv", help="The path to the input CSV file with consolidated spans.")
    parser.add_argument("output_csv", help="The path to save the labeled output CSV file.")
    
    args = parser.parse_args()
    
    labeler = CSVLabeler(args.input_csv, args.output_csv)
    labeler.run()