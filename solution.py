import sys
import time

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 07/10/2026
"""


class Solver:

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        #
        # TODO: Define any class instance variables you require (e.g. a dictionary mapping (state, action) to Q-value,
        #  or the weights of your linear Q-function) here.
        #
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Training rules (apply to both algorithms) ====================================================================
    #
    # Before training starts, the tester and autograder give self.game_env a training budget: a maximum number of
    # perform_action calls, and a lowest cumulative training reward. Once either runs out, perform_action raises
    # TrainingBudgetExhausted, which ends training - do not catch it. self.game_env.training_steps_used,
    # self.game_env.training_step_budget and self.game_env.training_reward_total let you plan around the budget.
    #
    # During training, perform_action must be called on either the initial state (self.game_env.get_init_state(), which
    # starts a new episode) or the state returned by your previous perform_action call.
    #
    # Your agent must learn from experience. The movement probabilities, action costs and penalties are hidden: each
    # testcase draws its own from its seed, and the environment keeps them private. Your agent may use the map (e.g.
    # distances, or what lies in an action's direction) and the rewards it receives, but must not read the
    # environment's private parameters or call its private methods, or plan with a transition model (e.g. value
    # iteration).
    #
    # Only some actions are valid on each tile (self.game_env.get_valid_actions(state)). An invalid action leaves the
    # state unchanged and still costs its action cost.

    # === Tabular Q-learning ===========================================================================================

    def ql_initialise(self):
        """
        Initialise any variables required before the start of tabular Q-learning.
        """
        self.Q = {}
        self.N = {}
        self.lava = set(self.game_env.lava_positions)   
        self.launch = set(self.game_env.launch_positions)
        self.valid = [[self.game_env.get_valid_actions(GameState(r, c, ())) for c in range(self.game_env.n_cols)]
                      for r in range(self.game_env.n_rows)]
        #
        # TODO: Implement any initialisation for tabular Q-learning (e.g. creating your Q-table) here. This method is
        #  not timed, but it must not call perform_action (training belongs in ql_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        progress = self.game_env.training_steps_used / self.game_env.training_step_budget
        self.epsilon = max(0.02, 0.1 - 0.16 * progress)
        s = self.game_env.get_init_state()
        steps = 0
        while not (self.game_env.is_game_over(s) or self.game_env.is_solved(s)) and steps < self.game_env.max_episode_steps:
            action_set = self.valid[s.row][s.col]
            best = max(self.Q.get((s, a), 0.0) for a in action_set)
            best_actions = [a for a in action_set if self.Q.get((s, a), 0.0) == best]
            best_a = random.choice(best_actions)
            if self.Q.get((s, best_a), 0.0) > best:
                best = self.Q.get((s, best_a), 0.0)
            if random.random() < self.epsilon:
                best_a = random.choice(action_set)
            s2, r, err_msg = self.game_env.perform_action(s, best_a)
            if self.game_env.is_game_over(s2) or self.game_env.is_solved(s2):
                best_next = 0.0
            else:
                best_next = max(self.Q.get((s2, a2), 0.0) for a2 in self.valid[s2.row][s2.col])
            n = self.N.get((s, best_a), 0) + 1
            self.N[(s, best_a)] = n
            alpha = max(0.1, 1 / n ** 0.5)
            td = r + self.game_env.gamma*best_next - self.Q.get((s, best_a), 0.0)
            self.Q[(s, best_a)] = self.Q.get((s, best_a), 0.0) + td*alpha
            s = s2
            steps += 1

            
        
        
        #
        # TODO: Implement a single training episode here: start from self.game_env.get_init_state() and repeatedly
        #  choose an action (e.g. epsilon-greedy), call self.game_env.perform_action and update your Q-values, until
        #  the episode ends (a terminal state, or a step limit such as self.game_env.max_episode_steps).
        #
        # The tester and autograder call this method repeatedly until the training budget is exhausted, and check how
        # well your greedy policy performs at regular checkpoints in between.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def ql_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned Q-values.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        s = state
        action_set = self.valid[s.row][s.col]
        best = max(self.Q.get((s, a), 0.0) for a in action_set)
        best_actions = [a for a in action_set if self.Q.get((s, a), 0.0) == best]
        best_a = random.choice(best_actions)
        

        return best_a
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned Q-values) here.
        #  This method is used to evaluate your agent (including during training), so it should not explore or update
        #  your Q-values.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Feature-based Q-learning =====================================================================================
    FQ_RISK_WEIGHTS = (0, 25, 100)   
    FQ_SIDEWAYS = {'l': 'ud', 'r': 'ud', 'u': 'lr', 'd': 'lr'}    
    FQ_DELTAS = {'l': (0, -1), 'r': (0, 1), 'u': (-1, 0), 'd': (1, 0)}
    def fq_initialise(self):
        """
        Initialise any variables required before the start of feature-based Q-learning.
        """
        #
        self.valid = [[self.game_env.get_valid_actions(GameState(r, c, ())) for c in range(self.game_env.n_cols)]
                      for r in range(self.game_env.n_rows)]
        self.lava = set(self.game_env.lava_positions)   
        self.launch = set(self.game_env.launch_positions)
        landmarks = list(self.game_env.crystal_positions) + list(self.game_env.launch_positions)
        start = self.game_env.get_init_state()
        self.fq_h = [self.CostToGo(self.game_env, {p: self.fq_costs_to(p, lam) for p in landmarks}) for lam in self.FQ_RISK_WEIGHTS]
        self.fq_worst = []  
        self.fq_cache = {}  
        for h in self.fq_h:
            costs = [h.compute_heuristic(GameState(r, c, start.crystal_status)) for r in range(self.game_env.n_rows) for c in range(self.game_env.n_cols) if self.game_env.grid_data[r][c] != self.game_env.ROCK_TILE]
            finite = [v for v in costs if v != float('inf')]
            self.fq_worst.append(max(finite) if finite else 0.0)
        self.w = np.zeros(len(self.fq_get_features(start, self.game_env.ACTIONS[0])))
        #
        pass

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        #
        # TODO: Implement your feature design here. Features that describe the situation (e.g. distances, risks) rather
        #  than identify the exact state are what let your agent act well in states it never visited during training.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        row, col, status = state.row, state.col, state.crystal_status
        kind, direction = action[0], action[1]
        distances = (0, 1, 2, 3, 4) if kind == 'b' else (1,)
        n = len(distances)
        cost = [0.0] * len(self.FQ_RISK_WEIGHTS)
        lava = solved = 0.0
        for distance in distances:
            lr, lc, _ = self.fq_slide(row, col, direction, distance)
            costs, in_lava, done = self.fq_landing(lr, lc, status)
            cost = [a + b / n for a, b in zip(cost, costs)]
            lava += in_lava / n
            solved += done / n
        drift_lava = 0.0
        for side in self.FQ_SIDEWAYS[direction]:
            for distance in distances:
                if self.fq_slide(row, col, side, distance)[:2] in self.lava:
                    drift_lava += 1.0 / (2 * n)
        double_lava = 0.0
        if kind != "b":
            lr, lc, bumped = self.fq_slide(row, col, direction, 1)
            if not bumped and (lr, lc) not in self.lava and self.game_env.grid_data[lr][lc] != '*':
                double_lava = 1.0 if self.fq_slide(lr, lc, direction, 1)[:2] in self.lava else 0.0
        bumped = 1.0 if self.fq_slide(row, col, direction, 1)[2] else 0.
        drift_cost = [0.0] * len(self.FQ_RISK_WEIGHTS)
        for side in self.FQ_SIDEWAYS[direction]:
            for distance in distances:
                lr, lc, _ = self.fq_slide(row, col, side, distance)
                costs, _, _ = self.fq_landing(lr, lc, status)
                drift_cost = [a + b / (2 * n) for a, b in zip(drift_cost, costs)]
        return [1.0] + cost + [lava, drift_lava, double_lava, solved, bumped] + drift_cost
        pass

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        progress = self.game_env.training_steps_used / self.game_env.training_step_budget
        self.epsilon = max(0.02, 0.1 - 0.16 * progress)
        alpha = 0.1 - 0.09 * progress 
        s = self.game_env.get_init_state()
        steps = 0
        while not (self.game_env.is_game_over(s) or self.game_env.is_solved(s)) and steps < self.game_env.max_episode_steps:
            action_set, phi = self.fq_state((s.row, s.col, s.crystal_status))
            q = phi @ self.w
            best = int(np.argmax(q))
            if random.random() < self.epsilon:
                best = random.randrange(len(action_set))
            s2, r, err_msg = self.game_env.perform_action(s, action_set[best])
            if self.game_env.is_game_over(s2) or self.game_env.is_solved(s2):
                best_next = 0.0
            else:
                best_next = float(np.max(self.fq_state((s2.row, s2.col, s2.crystal_status))[1] @ self.w))
            target = r + self.game_env.gamma * best_next
            target = min(0.0, max(-3000.0, target))
            td = target - q[best]
            self.w += alpha * td / float(phi[best] @ phi[best]) * phi[best]
            steps += 1
            s = s2
            
        #
        # TODO: Implement a single training episode here, as for ql_train_episode, but updating the weights of your
        #  linear Q-function instead of a Q-table.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        action_set, phi = self.fq_state((state.row, state.col, state.crystal_status))
        return action_set[int(np.argmax(phi @ self.w))]
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned weights) here.
        #  This method is also evaluated from start positions your agent never trained from.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    def fq_state(self, key):
        entry = self.fq_cache.get(key)
        if entry is None:
            row, col, status = key
            action_set = self.valid[row][col]
            state = GameState(row, col, status)
            entry = (action_set, np.array([self.fq_get_features(state, a) for a in action_set]))
            self.fq_cache[key] = entry
        return entry
    
    def fq_slide(self, row, col, direction, distance):
        delta_row, delta_col = self.FQ_DELTAS[direction]
        for _ in range(distance):
            candidate_row = row + delta_row
            candidate_col = col + delta_col
            if not self.fq_free(candidate_row, candidate_col):
                return row, col, True
            row, col = candidate_row, candidate_col
            if self.game_env.grid_data[candidate_row][candidate_col] == self.game_env.CRATER_TILE:
                break
            if self.game_env.grid_data[candidate_row][candidate_col] == self.game_env.LAVA_TILE:
                break
        return row, col, False
            
    def fq_landing(self, row, col, status):
        if (row, col) in self.lava:
            return [w / 10 for w in self.fq_worst], 1.0, 0.0
        if (row, col) in self.game_env.crystal_positions:
            crystal_index = self.game_env.crystal_positions.index((row, col))
            if status[crystal_index] == 0:
                status = list(status)
                status[crystal_index] = 1
                status = tuple(status)
        if (row, col) in self.launch and sum(status) >= self.game_env.min_samples:
            return [0.0] * len(self.FQ_RISK_WEIGHTS), 0.0, 1.0
        out = []
        for i, h in enumerate(self.fq_h):
            v = h.compute_heuristic(GameState(row, col, status)) 
            out.append((self.fq_worst[i] if v == float('inf') else v) / 10)
        return out, 0.0, 0.0
                
    #
    def fq_move_risk(self, row, col, direction):
        risk = 0.0
        for sides in self.FQ_SIDEWAYS[direction]:
            dr, dc = self.FQ_DELTAS[sides]
            if (row + dr, col + dc) in self.lava:
                risk += 0.5
        dr, dc = self.FQ_DELTAS[direction]
        if self.fq_free(row + dr, col + dc) and self.game_env.grid_data[row + dr][col + dc] != '*' and (row + 2*dr, col + 2*dc) in self.lava and self.game_env.grid_data[row][col] != '*':
            risk += 1
        return risk
    
    def fq_free(self, row, col):
        return 0 <= row < self.game_env.n_rows and 0 <= col < self.game_env.n_cols and self.game_env.grid_data[row][col] != self.game_env.ROCK_TILE
        
    def fq_costs_to(self, target, lam):
        dist = {target: 0.0}
        counter = 0
        frontier = [(0.0, counter, target)]
        while frontier:
            d, _, (row, col) = heapq.heappop(frontier)
            if d > dist.get((row, col), float("inf")):
                continue
            for direction, (dr, dc) in self.FQ_DELTAS.items():
                pred = (row - dr, col - dc)         
                if not self.fq_free(pred[0], pred[1]) or pred in self.lava:
                    continue
                nd = d + 1.0 + lam * self.fq_move_risk(pred[0], pred[1], direction)
                if nd >= dist.get(pred, float("inf")):
                    continue
                dist[pred] = nd
                counter += 1
                heapq.heappush(frontier, (nd, counter, pred))
        return dist
    
    class CostToGo:
        def __init__(self, game_env, dist_to):
            self.game_env = game_env
            self.crystals = self.game_env.crystal_positions
            self.launch = self.game_env.launch_positions
            self._dist_to = dist_to
            self._suffix_cache = {}
            self._h_cache = {}
            self.crystal_bank = {}
            self.nearest_launch = {}
            for crystal1 in self.crystals:
                for crystal2 in [c for c in self.crystals if c != crystal1]:
                    self.crystal_bank[(crystal1, crystal2)] = self._dist_to[crystal2].get(crystal1, float("inf"))
                self.nearest_launch[crystal1] = min(self._dist_to[l].get(crystal1, float("inf")) for l in self.launch)
            
            
            
        def _combine_with_position(self, suffix, state):
            best = float("inf")
            for c1, tail in suffix.items():
                best = min(best, self._pos_to_landmark(state, c1) + tail)
            return best
        
        def _pos_to_landmark(self, state, landmark):
            return self._dist_to[landmark].get((state.row, state.col), float("inf"))
    
        
        def _crystal_suffix(self, status, k):
            key = (status, k)
            cached = self._suffix_cache.get(key)
            if cached is not None:
                return cached
            uncollected = tuple(c for c, got in zip(self.crystals, status) if not got)
        
            if k == 1:
                result = {c1: self.nearest_launch[c1] for c1 in uncollected}
            elif k == 2:
                result = {
                    c1: min(self.crystal_bank[(c1, c2)] + self.nearest_launch[c2]
                             for c2 in uncollected if c2 != c1)
                    for c1 in uncollected
                }
            elif k == 3:
                result = {}
                for c1 in uncollected:
                    best = float("inf")
                    for c2 in uncollected:
                        if c2 == c1:
                            continue
                        for c3 in uncollected:
                            if c3 == c1 or c3 == c2:
                                continue
                            best = min(best, self.crystal_bank[(c1, c2)] + self.crystal_bank[(c2, c3)] + self.nearest_launch[c3])
                    result[c1] = best
            else:
                memo = {}
                def best_from(remaining_steps, current, available):
                    if remaining_steps == 0:
                        return self.nearest_launch[current]
                    mkey = (remaining_steps, current, available)
                    if mkey in memo:
                        return memo[mkey]
                    best = min(
                        self.crystal_bank[(current, nxt)] + best_from(remaining_steps - 1, nxt, available - {nxt})
                        for nxt in available
                    )
                    memo[mkey] = best
                    return best
                all_uncollected = frozenset(uncollected)
                result = {c1: best_from(k - 1, c1, all_uncollected - {c1}) for c1 in uncollected}
        
            self._suffix_cache[key] = result
            return result
        
        def compute_heuristic(self, state):
            h = self._h_cache.get(state)
            if h is not None:
                return h
            status = state.crystal_status
            remaining = self.game_env.min_samples - sum(status)
            if remaining <= 0:
                best = min(self._pos_to_landmark(state, l) for l in self.launch)
            else:
                suffix = self._crystal_suffix(status, remaining)
                best = self._combine_with_position(suffix, state)
            self._h_cache[state] = best
            return best 

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
