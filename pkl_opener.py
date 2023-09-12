import pickle

# Open and load the pickle file
with open('datasets/coil-100/processed/data_fold_4.pkl', 'rb') as f:
    data = pickle.load(f)

# Print the data to see its structure
print(data)
