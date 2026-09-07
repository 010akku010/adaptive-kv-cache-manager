class DecisionEngine:

    def calculate_utility(self, benefit, cost):
        return benefit - cost

    def decide(self, actions):
        best_action = None
        best_utility = float("-inf")

        for action, values in actions.items():
            benefit = values["benefit"]
            cost = values["cost"]

            utility = self.calculate_utility(benefit, cost)

            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action, best_utility