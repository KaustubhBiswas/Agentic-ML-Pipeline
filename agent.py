import pandas as pd
from predict import train_model, classify_values 
from parser import parse_phone_number, parse_company_name
import argparse

class Agent:
    def __init__(self, confidence_threshold=0.7):
        self.model, self.vectorizer = train_model()
        self.confidence_threshold = confidence_threshold

    def process(self, input_file, output_file="agent_output.csv"):
        df = pd.read_csv(input_file)
        results = []

        for col in df.columns:
            values = df[col].dropna().astype(str).tolist()

            label, conf = classify_values(values, self.model, self.vectorizer)
            print(f"Column: {col}, Predicted: {label}, Confidence: {conf:.2f}")

            
            if conf < self.confidence_threshold:
                print(f"[Fallback] Low confidence for column {col}, flagging for review.")
                
                continue

            
            for idx, value in df[col].dropna().items():
                value_str = str(value)
                if label == "Phone Number":
                    country, number = parse_phone_number(value_str)
                    results.append({
                        'Column': col, 'Type': label, 'Value': value_str,
                        'Country': country, 'Number': number, 'Confidence': conf
                    })
                elif label == "Company Name":
                    name, legal = parse_company_name(value_str)
                    results.append({
                        'Column': col, 'Type': label, 'Value': value_str,
                        'Name': name, 'Legal': legal, 'Confidence': conf
                    })
                else:
                    results.append({
                        'Column': col, 'Type': label, 'Value': value_str,
                        'Confidence': conf
                    })

        
        if results:
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"✅ Results saved to {output_file}")
        else:
            print("⚠️ No classified data found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent orchestrating Part A (predict) + Part B (parser)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="agent_output.csv", help="Path to output CSV file")
    args = parser.parse_args()

    agent = Agent(confidence_threshold=0.6)
    agent.process(args.input, args.output)

