import pandas as pd

class ML_test:
    def __init__ (self):
        self.best_results = []
        self.round_results = []
        self.round_subsets = []
        self.history = {}
        self.n_tests = 0
        self.expanded = []
        
    def save_history(self, filename="history.csv"):
        df = pd.DataFrame()
        df["history"] = self.history.keys()
        print "total explored: ", len(self.history)
        df.to_csv(filename, index=False)
        
    def save_scores(self, filename="results.csv"):
        df = pd.DataFrame()
        df["scores"] = self.best_results
        df.to_csv(filename, index=False)
    
        
    def save_expanded(self, filename="history_expanded.csv"):
        df = pd.DataFrame()
        df["history_expanded"] = self.expanded
        df.to_csv(filename, index=False)