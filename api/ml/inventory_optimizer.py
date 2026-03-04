"""
Inventory Optimization using Gurobi
Minimizes total costs while maintaining service levels
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not available. Install with: pip install gurobipy")


@dataclass
class Product:
    """Product information for inventory optimization"""
    sku: str
    name: str
    unit_cost: float
    holding_cost_rate: float
    ordering_cost: float
    demand_mean: float
    demand_std: float
    lead_time: int
    service_level: float = 0.95


class InventoryOptimizer:
    """Multi-product inventory optimization using Gurobi"""
    
    def __init__(self):
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is required")
        self.model = None
        self.products = []
        self.results = {}
        
    def add_product(self, product: Product):
        self.products.append(product)
        
    def calculate_safety_stock(
        self, 
        demand_std: float, 
        lead_time: int,
        service_level: float
    ) -> float:
        from scipy import stats
        z = stats.norm.ppf(service_level)
        safety_stock = z * demand_std * np.sqrt(lead_time)
        return safety_stock
    
    def optimize(
        self,
        budget_constraint: Optional[float] = None,
        storage_constraint: Optional[float] = None,
        time_limit: int = 300
    ) -> Dict:
        if not self.products:
            raise ValueError("No products added")
        
        model = gp.Model("inventory_optimization")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)
        
        order_qty = {}
        reorder_point = {}
        safety_stock = {}
        
        for product in self.products:
            sku = product.sku
            order_qty[sku] = model.addVar(lb=0, name=f"order_qty_{sku}")
            reorder_point[sku] = model.addVar(lb=0, name=f"reorder_point_{sku}")
            safety_stock[sku] = model.addVar(lb=0, name=f"safety_stock_{sku}")
        
        total_cost = 0
        
        for product in self.products:
            sku = product.sku
            annual_demand = product.demand_mean * 365
            annual_holding_cost = product.unit_cost * product.holding_cost_rate
            ordering_cost = (annual_demand / order_qty[sku]) * product.ordering_cost
            avg_inventory = order_qty[sku] / 2 + safety_stock[sku]
            holding_cost = avg_inventory * annual_holding_cost
            total_cost += ordering_cost + holding_cost
        
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        for product in self.products:
            sku = product.sku
            required_safety_stock = self.calculate_safety_stock(
                product.demand_std, product.lead_time, product.service_level
            )
            model.addConstr(safety_stock[sku] >= required_safety_stock)
            lead_time_demand = product.demand_mean * product.lead_time
            model.addConstr(reorder_point[sku] == lead_time_demand + safety_stock[sku])
        
        if budget_constraint:
            total_inventory_value = sum(
                (order_qty[p.sku] / 2 + safety_stock[p.sku]) * p.unit_cost
                for p in self.products
            )
            model.addConstr(total_inventory_value <= budget_constraint)
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            results = {'status': 'optimal', 'total_cost': model.objVal, 'products': {}}
            
            for product in self.products:
                sku = product.sku
                annual_demand = product.demand_mean * 365
                annual_holding_cost = product.unit_cost * product.holding_cost_rate
                
                q = order_qty[sku].X
                r = reorder_point[sku].X
                s = safety_stock[sku].X
                
                results['products'][sku] = {
                    'name': product.name,
                    'order_quantity': round(q, 2),
                    'reorder_point': round(r, 2),
                    'safety_stock': round(s, 2),
                    'average_inventory': round(q / 2 + s, 2)
                }
            
            naive_cost = self._calculate_naive_cost()
            savings = naive_cost - model.objVal
            
            results['cost_savings'] = {
                'naive_cost': round(naive_cost, 2),
                'optimized_cost': round(model.objVal, 2),
                'savings_amount': round(savings, 2),
                'savings_percent': round((savings / naive_cost) * 100, 2)
            }
        else:
            results = {'status': 'infeasible'}
        
        self.results = results
        return results
    
    def _calculate_naive_cost(self) -> float:
        total_cost = 0
        for product in self.products:
            annual_demand = product.demand_mean * 365
            annual_holding_cost = product.unit_cost * product.holding_cost_rate
            naive_order_qty = annual_demand / 12
            naive_safety_stock = product.demand_mean * product.lead_time * 2
            ordering_cost = 12 * product.ordering_cost
            holding_cost = (naive_order_qty / 2 + naive_safety_stock) * annual_holding_cost
            total_cost += ordering_cost + holding_cost
        return total_cost


if __name__ == "__main__":
    optimizer = InventoryOptimizer()
    
    products = [
        Product("SKU-001", "Laptop", 1200, 0.20, 50, 10, 3, 7),
        Product("SKU-002", "Monitor", 400, 0.18, 30, 15, 5, 5),
        Product("SKU-003", "Mouse", 25, 0.15, 20, 50, 10, 3)
    ]
    
    for p in products:
        optimizer.add_product(p)
    
    results = optimizer.optimize(budget_constraint=100000)
    print(f"Total Cost: ${results['total_cost']:,.2f}")
    print(f"Savings: {results['cost_savings']['savings_percent']}%")
