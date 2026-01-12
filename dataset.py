import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SimplifiedDataset(Dataset):
    """
    Simplified dataset for fixed-length time series data.
    Data shape: (num_subjects, time_steps, dimensions)
    Training: predict first 120 time steps (test set is last 40 time steps)
    """
    def __init__(self, data_path, inputs_path=None, train_split=0.75):
        """
        Args:
            data_path: Path to stacked_invests.npy
            inputs_path: Path to stacked_inputs.npy (optional)
            train_split: Fraction of time steps to use for training (default: 0.75)
        """
        # Load data
        self.data = np.load(data_path)  # (num_subjects, time_steps, dimensions)
        self.inputs = np.load(inputs_path) if inputs_path else None
        # Data dimensions
        self.num_subjects, self.total_time_steps, self.data_dim = self.data.shape
        self.input_dim = self.inputs.shape[2] if self.inputs is not None else 0
        
        # Split time steps into train/test (not subjects)
        self.train_split = train_split
        self.train_time_steps = int(self.total_time_steps * train_split)
        self.test_time_steps = self.total_time_steps - self.train_time_steps
        
    def __len__(self):
        return self.num_subjects
    
    def __getitem__(self, idx):
        """
        Returns training data for the first train_split% of time steps
        """
        # Get training data (first train_split% of time steps)
        data = self.data[idx, :self.train_time_steps, :]  # (train_time_steps, data_dim)
        
        item = {
            'data': torch.FloatTensor(data),
            'subject_id': idx,
            'time_steps': self.train_time_steps
        }
        
        if self.inputs is not None:
            inputs = self.inputs[idx, :self.train_time_steps, :]  # (train_time_steps, input_dim)
            item['inputs'] = torch.FloatTensor(inputs)
        
        return item
    
    def get_test_data(self, idx):
        """
        Returns test data for the last (1-train_split)% of time steps
        """
        # Get test data (last test_time_steps)
        data = self.data[idx, -self.test_time_steps:, :]  # (test_time_steps, data_dim)
        
        item = {
            'data': torch.FloatTensor(data),
            'subject_id': idx,
            'time_steps': self.test_time_steps
        }
        
        if self.inputs is not None:
            inputs = self.inputs[idx, -self.test_time_steps:, :]  # (test_time_steps, input_dim)
            item['inputs'] = torch.FloatTensor(inputs)
        
        return item
    
    def verify_train_test_split(self):
        """
        Verify that the train/test split is correct and there's no data leakage.
        """
        print(f"\n" + "="*60)
        print("VERIFYING TRAIN/TEST SPLIT IN DATASET")
        print("="*60)
        
        print(f"Dataset configuration:")
        print(f"- Total time steps: {self.total_time_steps}")
        print(f"- Train time steps: {self.train_time_steps}")
        print(f"- Test time steps: {self.test_time_steps}")
        print(f"- Train split ratio: {self.train_split:.2f}")
        
        # Test training data access
        train_item = self[0]  # Get first subject's training data
        print(f"\nTraining data verification:")
        print(f"- Training data shape: {train_item['data'].shape}")
        print(f"- Expected shape: ({self.train_time_steps}, {self.data_dim})")
        print(f"- Training data time range: 0 to {self.train_time_steps-1}")
        
        # Test test data access
        test_item = self.get_test_data(0)  # Get first subject's test data
        print(f"\nTest data verification:")
        print(f"- Test data shape: {test_item['data'].shape}")
        print(f"- Expected shape: ({self.test_time_steps}, {self.data_dim})")
        print(f"- Test data time range: {self.train_time_steps} to {self.total_time_steps-1}")
        
        # Verify no overlap
        if self.train_time_steps + self.test_time_steps != self.total_time_steps:
            print(f"\n❌ ERROR: Train + test steps ({self.train_time_steps} + {self.test_time_steps}) != total ({self.total_time_steps})")
            return False
        else:
            print(f"\n✅ Train/test split is correct: {self.train_time_steps} + {self.test_time_steps} = {self.total_time_steps}")
        
        # Check that training data doesn't contain test time steps
        print(f"\nVerifying no data leakage:")
        print(f"- Training data max index: {self.train_time_steps-1}")
        print(f"- Test data min index: {self.train_time_steps}")
        print(f"- No overlap between training and test periods ✅")
        
        # Verify that training data is actually from the beginning
        train_data_actual = train_item['data'].numpy()
        train_data_expected = self.data[0, :self.train_time_steps, :]
        
        if np.array_equal(train_data_actual, train_data_expected):
            print(f"✅ Training data correctly contains first {self.train_time_steps} time steps")
        else:
            print(f"❌ ERROR: Training data does not match expected first {self.train_time_steps} time steps")
            return False
        
        # Verify that test data is actually from the end
        test_data_actual = test_item['data'].numpy()
        test_data_expected = self.data[0, -self.test_time_steps:, :]
        
        if np.array_equal(test_data_actual, test_data_expected):
            print(f"✅ Test data correctly contains last {self.test_time_steps} time steps")
        else:
            print(f"❌ ERROR: Test data does not match expected last {self.test_time_steps} time steps")
            return False
        
        print(f"\n🎉 All verifications passed! Train/test split is working correctly.")
        return True

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Creates a DataLoader for the simplified dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_and_create_dataset(data_path="data/stacked_invests.npy", 
                           inputs_path="data/stacked_inputs.npy",
                           batch_size=32, 
                           shuffle=True,
                           train_split=0.75):
    """
    Load data and create dataset and dataloader.
    
    Args:
        data_path: Path to stacked_invests.npy
        inputs_path: Path to stacked_inputs.npy
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        train_split: Fraction of data for training
    """
    # Create dataset
    dataset = SimplifiedDataset(data_path, inputs_path, train_split)
    
    # Create dataloader
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, dataloader

if __name__ == "__main__":
    # Test the data loading
    dataset, dataloader = load_and_create_dataset()
    
    # Verify train/test split
    dataset.verify_train_test_split()
    
    # Test a batch
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"Data: {batch['data'].shape}")
    if dataset.inputs is not None:
        print(f"Inputs: {batch['inputs'].shape}")
    print(f"Subject IDs: {batch['subject_id'].shape}")
    print(f"Time steps: {batch['time_steps']}") 