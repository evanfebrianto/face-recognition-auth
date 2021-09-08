import csv
import pickle

class Reset:
    def __init__(self):
        self.inventory_name = ['Cabinet 1', 'Cabinet 2', 'Cabinet 3', 'Cabinet 4', 'Cabinet 5']
        self.inventory_count = [10, 10, 10, 10, 10]
        self.default_personal_count = 0
        self.identities = None
        self.inventories = None
        self.inventory_current_count = None
        self.load_pickle()
        
    def _initialize_inventory(self):
        self.load_pickle()
        self.reset_pickle()
        self.reset_csv_logs()

    def load_pickle(self):
        with open('identities.pkl', 'rb') as f:
            x = pickle.load(f)
        self.identities = list(x.keys())
        
        with open('inventories.pkl', 'rb') as f:
            y = pickle.load(f)
        self.inventories = y.copy()
        self.inventory_current_count = list(y['Inventory'].values())
        # self.inventory_count = self.inventory_current_count.copy()

    def save_pickle(self, inventory):
        with open('inventories.pkl', 'wb') as handle:
            pickle.dump(inventory, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset_csv_logs(self):
        filepath = './report.csv'
        with open(filepath, "w+") as file:
            writer = csv.writer(file, lineterminator='')
            writer.writerow(['timestamp', 'PIC', 'logs'])
        

    def reset_pickle(self):
        self.inventory_current_count = self.inventory_count.copy()
        self.inventories = {'Inventory':{}}
        for i, (name, count) in enumerate(zip(self.inventory_name, self.inventory_count)):
            self.inventories['Inventory'].update({name : count})
            
        for identity in self.identities:
            self.inventories.update({identity : {}})
            for i, name in enumerate(self.inventory_name):
                self.inventories[identity].update({name : self.default_personal_count})
        
        # self.inventories = inventories
        self.save_pickle(self.inventories)

    def store_pickle(self, count_inventory, count_belongings, person_name):
        # inventories = {'Inventory':{}}
        # print(count_belongings, person_name)
        for i, (inventory_name, inventory) in enumerate(zip(self.inventory_name, count_inventory)):
            self.inventories['Inventory'].update({inventory_name : inventory})
        for i, (stuff_name, stuff) in enumerate(zip(self.inventory_name, count_belongings)):
            self.inventories[person_name].update({stuff_name : stuff})
        self.save_pickle(self.inventories)

def main():
    r = Reset()
    r._initialize_inventory()
    # print(r.inventory_current_count)

if __name__ == '__main__':
    main()